from abc import abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Any, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.event import Event, Bar, Quote
from roboquant.strategies.buffer import OHLCVBuffer

T = TypeVar("T")


class Feature(Generic[T]):
    """Base class for all types of features"""

    @abstractmethod
    def calc(self, value: T) -> NDArray:
        """
        Perform the calculation and return the result as a 1-dimensional NDArray.
        The result should always be the same size. If a value cannot be calculated at a certain
        time, it should use a float NaN in the NDArray.
        """

    @abstractmethod
    def size(self) -> int:
        """return the size of this feature"""

    def reset(self):
        """Reset the state of the feature"""

    def _shape(self):
        """return the shape of this feature as a tuple"""
        return (self.size(),)

    def _zeros(self):
        return np.zeros(self._shape(), dtype=np.float32)

    def _ones(self):
        return np.ones(self._shape(), dtype=np.float32)

    def _full_nan(self):
        return np.full(self._shape(), float("nan"), dtype=np.float32)

    def cache(self, validate=False) -> "Feature[T]":
        return CacheFeature(self, validate)

    def returns(self, period=1) -> "Feature[T]":
        if period == 1:
            return ReturnFeature(self)  # type: ignore
        return LongReturnsFeature(self, period)  # type: ignore

    def normalize(self, min_period=3) -> "Feature[T]":
        return NormalizeFeature(self, min_period)  # type: ignore

    def __getitem__(self, *args) -> "Feature[T]":
        return SlicedFeature(self, args)  # type: ignore


class EquityFeature(Feature[Account]):
    """Calculates the total equity value of the account"""

    def calc(self, value):
        return np.array(value.equity_value(), dtype=np.float32)

    def size(self):
        return 1


class SlicedFeature(Feature):
    """Calculate a slice from another feature"""

    def __init__(self, feature: Feature, args) -> None:
        super().__init__()
        self.args = args
        self.feature = feature
        self._size = len(feature._zeros()[args])

    def calc(self, value):
        values = self.feature.calc(value)
        return values[self.args]

    def size(self):
        return self._size

    def reset(self):
        return self.feature.reset()


class TrueRangeFeature(Feature[Event]):
    """Calculates the true range value for a single asset that has a Bar price item (candlestick) in the event"""

    def __init__(self, asset: Asset) -> None:
        super().__init__()
        self.prev_close = None
        self.asset = asset

    def calc(self, value: Event):
        item = value.price_items.get(self.asset)
        if item is None or not isinstance(item, Bar):
            return np.array([float("nan")])

        ohlcv = item.ohlcv
        high = ohlcv[1]
        low = ohlcv[2]
        close = ohlcv[3]

        prev_close = self.prev_close if self.prev_close is not None else low
        self.prev_close = close

        result = max(high - low, abs(high - prev_close), abs(low - prev_close))

        return np.array([result], dtype=np.float32)

    def size(self) -> int:
        return 1

    def reset(self):
        self.prev_close = None


class FixedValueFeature(Feature):

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = np.array(value, dtype="float32")

    def size(self) -> int:
        return len(self.value)

    def calc(self, value):
        return self.value


class PriceFeature(Feature[Event]):
    """Extract a single price for one or more assets in the event"""

    def __init__(self, *assets: Asset, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.assets = assets
        self.price_type = price_type

    def calc(self, value):
        prices = [value.get_price(asset, self.price_type) for asset in self.assets]
        return np.array(prices, dtype=np.float32)

    def size(self) -> int:
        return len(self.assets)


class BarFeature(Feature[Event]):
    """Extract the ohlcv values from bars for one or more assets"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, value):
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = value.price_items.get(asset)
            if isinstance(item, Bar):
                offset = idx * 5
                result[offset: offset + 5] = item.ohlcv

        return result

    def size(self) -> int:
        return 5 * len(self.assets)


class QuoteFeature(Feature[Event]):
    """Extract the values from quotes for one or more assets"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, value):
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = value.price_items.get(asset)
            if isinstance(item, Quote):
                offset = idx * 4
                result[offset: offset + 4] = item.data

        return result

    def size(self) -> int:
        return 4 * len(self.assets)


class CombinedFeature(Feature):
    """Combine multiple features into one single feature by stacking them."""

    def __init__(self, *features: Feature) -> None:
        super().__init__()
        self.features = features
        self._size = sum(feature.size() for feature in self.features)

    def calc(self, value):
        data = [feature.calc(value) for feature in self.features]
        return np.hstack(data, dtype=np.float32)

    def size(self) -> int:
        return self._size

    def reset(self):
        for feature in self.features:
            feature.reset()


class NormalizeFeature(Feature):
    """online normalization calculator"""

    def __init__(self, feature: Feature, min_count: int = 3) -> None:
        super().__init__()
        self.feature = feature
        self.min_count = min_count
        self.existing_aggregate = (self._zero_int(), self._zeros(), self._zeros())

    def _zero_int(self):
        return np.zeros((self.size(),), dtype="int")

    def denormalize(self, value):
        (count, mean, m2) = self.existing_aggregate
        stdev = np.sqrt(m2 / count) - 1e-12
        return value * stdev + mean

    def __update(self, new_value):
        (count, mean, m2) = self.existing_aggregate
        mask = ~np.isnan(new_value)
        count[mask] += 1
        delta = new_value - mean
        mean[mask] += delta[mask] / count[mask]
        delta2 = new_value - mean
        m2[mask] += delta[mask] * delta2[mask]

    def __normalize_values(self, values):
        (count, mean, m2) = self.existing_aggregate
        stdev = self._full_nan()
        mask = count >= self.min_count
        stdev[mask] = np.sqrt(m2[mask] / count[mask]) + 1e-12
        return (values - mean) / stdev

    def calc(self, value):
        values = self.feature.calc(value)
        self.__update(values)
        return self.__normalize_values(values)

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.existing_aggregate = (self._zero_int(), self._zeros(), self._zeros())
        self.feature.reset()


class FillFeature(Feature):
    """If a feature contains a NaN value, use the last known value instead"""

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = self._full_nan()

    def calc(self, value):
        values = self.feature.calc(value)
        mask = np.isnan(values)
        values[mask] = self.fill[mask]
        self.fill = np.copy(values)
        return values

    def reset(self):
        self.fill = self._full_nan()
        self.feature.reset()

    def size(self) -> int:
        return self.feature.size()


class FillWithConstantFeature(Feature):
    """If a feature contains a NaN value, fill with a constant value"""

    def __init__(self, feature: Feature, constant: float = 0.0) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = self._zeros()
        self.fill = np.full(self._shape(), constant, dtype=np.float32)

    def calc(self, value):
        values = self.feature.calc(value)
        mask = np.isnan(values)
        values[mask] = self.fill[mask]
        return values

    def reset(self):
        self.feature.reset()

    def size(self) -> int:
        return self.feature.size()


class CacheFeature(Feature):
    """Cache the results of a feature from a previous run. This can speed up the learning process a lot, but
    this requires that:

    - the feed to have always an increasing time value (monotonic)
    - the feature to produce the same output at a given time. Typically, this doesn't hold true for features that
    are based on account values.
    """

    def __init__(self, feature: Feature, validate=False) -> None:
        super().__init__()
        self.feature: Feature = feature
        self._cache: dict[datetime, NDArray] = {}
        self.validate = validate

    def calc(self, value):
        time = value.time
        if time in self._cache:
            values = self._cache[time]
            if self.validate:
                calc_values = self.feature.calc(value)
                assert np.array_equal(
                    values, calc_values, equal_nan=True
                ), f"Wrong cache time={time} cache={values} calculated={calc_values}"
            return values

        values = self.feature.calc(value)
        self._cache[time] = values
        return values

    def reset(self):
        """Reset the underlying feature. This doesn't clear the cache"""
        self.feature.reset()

    def clear(self):
        """Clear all the cache"""
        self._cache = {}

    def size(self) -> int:
        return self.feature.size()


class VolumeFeature(Feature[Event]):
    """Extract the volume for one or more assets in the event"""

    def __init__(self, *assets: Asset, volume_type: str = "DEFAULT") -> None:
        super().__init__()
        self.assets = assets
        self.volume_type = volume_type

    def calc(self, value: Event):
        volumes = [value.get_volume(asset, self.volume_type) for asset in self.assets]
        return np.array(volumes, dtype=np.float32)

    def size(self) -> int:
        return len(self.assets)


class ReturnFeature(Feature):
    """Calculate the return of another feature"""

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.history = self._full_nan()

    def calc(self, value):
        values = self.feature.calc(value)
        r = values / self.history - 1.0
        self.history = values
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history = self._full_nan()
        self.feature.reset()


class LongReturnsFeature(Feature):

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value):
        values = self.feature.calc(value)
        h = self.history

        if len(h) < h.maxlen:  # type: ignore
            h.append(values)
            return self._full_nan()

        r = values / h[0] - 1.0
        h.append(values)
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history.clear()
        self.feature.reset()


class MaxReturnFeature(Feature):
    """Calculate the maximum return over a certain period.
    This will only work on features that return a single value.
    """

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        assert feature.size() == 1
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value):
        values = self.feature.calc(value)
        h = self.history
        h.append(values)

        if len(h) < h.maxlen:  # type: ignore
            return self._full_nan()

        r = max(h) / h[0] - 1.0
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history.clear()
        self.feature.reset()


class MinReturnFeature(Feature):
    """Calculate the minimum return over a certain period.
    This will only work for features that return a single value.
    """

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value):
        values = self.feature.calc(value)
        h = self.history
        h.append(values)

        if len(h) < h.maxlen:  # type: ignore
            return self._full_nan()

        r = min(h) / h[0] - 1.0
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history.clear()
        self.feature.reset()


class SMAFeature(Feature):
    """Calculate the simple moving average of another feature."""

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.period = period
        self.feature: Feature = feature
        self.history = None
        self._cnt = 0

    def calc(self, value):
        values = self.feature.calc(value)
        if self.history is None:
            self.history = np.zeros((self.period, values.size))

        idx = self._cnt % self.period
        self.history[idx] = values
        self._cnt += 1

        if self._cnt < self.period:
            return self._full_nan()

        return np.mean(self.history, axis=0)

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history = None
        self.feature.reset()
        self._cnt = 0


class DayOfWeekFeature(Feature[Event]):
    """Calculate a one-hot-encoded day of the week where Monday == 0 and Sunday == 6"""

    def __init__(self, tz=timezone.utc) -> None:
        super().__init__()
        self.tz = tz

    def calc(self, value: Event):
        dt = datetime.astimezone(value.time, self.tz)
        weekday = dt.weekday()
        result = np.zeros(7, dtype=np.float32)
        result[weekday] = 1.0
        return result

    def size(self) -> int:
        return 7


class TimeDifference(Feature[Event]):
    """Calculate the time difference in seconds between two consecutive events."""

    def __init__(self) -> None:
        super().__init__()
        self._last_time: datetime | None = None

    def calc(self, value: Event):
        if self._last_time:
            diff = value.time - self._last_time
            self._last_time = value.time
            return np.asarray([diff.total_seconds()], dtype="float32")

        self._last_time = value.time
        return self._full_nan()

    def size(self) -> int:
        return 1

    def reset(self):
        self._last_time = None


class TaFeature(Feature[Event]):
    """Base class for technical analysis features"""

    def __init__(self, *assets: Asset, history_size: int) -> None:
        super().__init__()
        self._data: dict[Asset, OHLCVBuffer] = {}
        self._size = history_size
        self.assets = list(assets)

    def calc(self, value: Event):
        result = []
        nan = float("nan")
        for asset in self.assets:
            v = nan
            item = value.price_items.get(asset)
            if isinstance(item, Bar):
                if asset not in self._data:
                    self._data[asset] = OHLCVBuffer(self._size)
                ohlcv = self._data[asset]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    v = self._calc(asset, ohlcv)

            result.append(v)
        return np.asarray(result, dtype=np.float32)

    @abstractmethod
    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        """Override this method with technical analysis logic"""
        ...

    def size(self) -> int:
        return len(self.assets)

    def reset(self):
        self._data = {}
