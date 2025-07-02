from abc import abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Any, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray, ArrayLike

from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.event import Event, Bar, Quote
from roboquant.strategies.buffer import OHLCVBuffer

T = TypeVar("T")
FloatArray = NDArray[np.float32]


class Feature(Generic[T]):
    """Base class for different types of features.
    The ones included by default are either based either an `Event` or an `Account`.
    Typically Event features are used for input and Account features are used for label/output."""

    @abstractmethod
    def calc(self, value: T) -> FloatArray:
        """
        Perform the calculation and return the result as a 1-dimensional NDArray of type float32.
        The result should always be the same size. If a value cannot be calculated at a certain
        time, it should use a float NaN in the NDArray.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the size of this feature"""

    def reset(self):
        """Reset the state of the feature"""

    def _shape(self):
        """return the shape of this feature as a tuple"""
        return (self.size(),)

    def _zeros(self):
        """Return a zero array of the correct shape"""
        return np.zeros(self._shape(), dtype=np.float32)

    def _ones(self):
        """Return an array of ones of the correct shape"""
        return np.ones(self._shape(), dtype=np.float32)

    def _full_nan(self) -> FloatArray:
        """Return a full NaN array of the correct shape"""
        return np.full(self._shape(), float("nan"), dtype=np.float32)

    def returns(self, period=1) -> "Feature[T]":
        if period == 1:
            return ReturnFeature(self)
        return LongReturnsFeature(self, period)

    def normalize(self, min_period=3) -> "Feature[T]":
        """Normalize the feature values by calculating the mean and standard deviation."""
        return NormalizeFeature(self, min_period)

    def __getitem__(self, *args) -> "Feature[T]":
        """Return a slice of the feature.
        For example, if the feature has size 10 and you call `feature[2:5]`, it will return a new feature
        that contains the values for indices 2, 3, and 4 of the original feature.
        """
        return SlicedFeature(self, args)


##############################
# Generic features
##############################


class SlicedFeature(Feature[T]):
    """Calculate a slice from another feature"""

    def __init__(self, feature: Feature[T], args) -> None:
        super().__init__()
        self.args = args
        self.feature = feature
        self._size = len(feature._zeros()[args])

    def calc(self, value: T) -> FloatArray:
        values = self.feature.calc(value)
        return values[self.args]

    def size(self):
        return self._size

    def reset(self):
        return self.feature.reset()


class FixedValueFeature(Feature):
    def __init__(self, value: ArrayLike) -> None:
        super().__init__()
        self.value = np.array(value, dtype="float32")

    def size(self) -> int:
        return len(self.value)

    def calc(self, value: Any) -> FloatArray:
        return self.value


class CombinedFeature(Feature[T]):
    """Combine multiple features into one single feature by horizontal stacking them.
    So if feature1 has size 3 and feature2 has size 5, the combined feature will have size 8.
    """

    def __init__(self, *features: Feature[T]) -> None:
        super().__init__()
        self.features = features
        self._size = sum(feature.size() for feature in self.features)

    def calc(self, value: T) -> FloatArray:
        data = [feature.calc(value) for feature in self.features]
        return np.hstack(data, dtype=np.float32)

    def size(self) -> int:
        return self._size

    def reset(self):
        for feature in self.features:
            feature.reset()


class NormalizeFeature(Feature[T]):
    """online normalization calculator"""

    def __init__(self, feature: Feature[T], min_count: int = 3) -> None:
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

    def __normalize_values(self, values) -> FloatArray:
        (count, mean, m2) = self.existing_aggregate
        stdev = self._full_nan()
        mask = count >= self.min_count
        stdev[mask] = np.sqrt(m2[mask] / count[mask]) + 1e-12  # type: ignore
        return (values - mean) / stdev

    def calc(self, value: T) -> FloatArray:
        values = self.feature.calc(value)
        self.__update(values)
        return self.__normalize_values(values)

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.existing_aggregate = (self._zero_int(), self._zeros(), self._zeros())
        self.feature.reset()


class FillFeature(Feature[T]):
    """If a feature contains a NaN value, use the last known value instead"""

    def __init__(self, feature: Feature[T]) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = self._full_nan()

    def calc(self, value: T) -> FloatArray:
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


class FillWithConstantFeature(Feature[T]):
    """If a feature contains a NaN value, fill with a constant value"""

    def __init__(self, feature: Feature[T], constant: float = 0.0) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = np.full(self._shape(), constant, dtype=np.float32)

    def calc(self, value: T) -> FloatArray:
        values = self.feature.calc(value)
        mask = np.isnan(values)
        values[mask] = self.fill[mask]
        return values

    def reset(self):
        self.feature.reset()

    def size(self) -> int:
        return self.feature.size()


class ReturnFeature(Feature[T]):
    """Calculate the return of another feature"""

    def __init__(self, feature: Feature[T]) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.history = self._full_nan()

    def calc(self, value: T) -> FloatArray:
        values = self.feature.calc(value)
        r: FloatArray = values / self.history - np.float32(1.0)  # type: ignore
        self.history = values
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history = self._full_nan()
        self.feature.reset()


class LongReturnsFeature(Feature[T]):
    def __init__(self, feature: Feature[T], period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value: T) -> FloatArray:
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


class MaxReturnFeature(Feature[T]):
    """Calculate the maximum return over a certain period.
    For now this will only work on features that return a single value.
    """

    def __init__(self, feature: Feature[T], period: int) -> None:
        super().__init__()
        assert feature.size() == 1
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value: T) -> FloatArray:
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


class MinReturnFeature(Feature[T]):
    """Calculate the minimum return over a certain period.
    This will only work for features that return a single value.
    """

    def __init__(self, feature: Feature[T], period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, value: T) -> FloatArray:
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


class SMAFeature(Feature[T]):
    """Calculate the simple moving average of another feature."""

    def __init__(self, feature: Feature[T], period: int) -> None:
        super().__init__()
        self.period = period
        self.feature: Feature = feature
        self.history = np.zeros((self.period, feature.size()), dtype=np.float32)
        self._cnt = 0

    def calc(self, value: T) -> FloatArray:
        values = self.feature.calc(value)
        idx = self._cnt % self.period
        self.history[idx] = values
        self._cnt += 1

        if self._cnt < self.period:
            return self._full_nan()

        return np.mean(self.history, axis=0)

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history = np.zeros((self.period, self.feature.size()), dtype=np.float32)
        self.feature.reset()
        self._cnt = 0


##############################
# Account features
##############################


class EquityFeature(Feature[Account]):
    """Calculates the total equity value of the account"""

    def calc(self, value: Account) -> FloatArray:
        return np.array(value.equity_value(), dtype=np.float32)

    def size(self):
        return 1


class UnrealizedPNLFeature(Feature[Account]):
    """Calculates the unrealized PNL % of all the open positions in the account
    If there are no open positions, this will return 0.0

    ```
        result = unrealized PNL / market value of all open positions
    ```
    """

    def calc(self, value: Account) -> FloatArray:
        mkt_value = value.convert(value.mkt_value())
        pnl = value.unrealized_pnl_value()
        if mkt_value and pnl:
             return np.array(pnl/mkt_value, dtype=np.float32)
        return np.array(0.0, dtype=np.float32)

    def size(self):
        return 1


#############################
# Event features
#############################


class DayOfWeekFeature(Feature[Event]):
    """Calculate a day of the week where Monday == 0 and Sunday == 6.
    The result can be one-hot encoded or not, depending on the `one_hot_encoded` parameter.
    If `one_hot_encoded` is True, the result will be a 7-element array
    else the result will be a single value representing the day of the week (0-6).
    For example, if the event time is a Monday, the result will be:
    - one-hot encoded: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    - not one-hot encoded: [0.0]
    """

    def __init__(self, tz=timezone.utc, one_hot_encoded: bool = True) -> None:
        super().__init__()
        self.tz = tz
        self.one_hot_encoded = one_hot_encoded

    def calc(self, value: Event) -> FloatArray:
        dt = datetime.astimezone(value.time, self.tz)
        weekday = dt.weekday()
        if not self.one_hot_encoded:
            return np.array([weekday], dtype=np.float32)

        result = np.zeros(7, dtype=np.float32)
        result[weekday] = 1.0
        return result

    def size(self) -> int:
        return 7 if self.one_hot_encoded else 1


class DayOfMonthFeature(Feature[Event]):
    """Calculate a day of month where the first day of the month is 0 and the last day is 30.
    Result can be one-hot encoded or not, depending on the `one_hot_encoded` parameter.
    If `one_hot_encoded` is True, the result will be a 30-element array else the result will be a single value.
    For example, if the event time is the 15th of the month, the result will be:
    - one-hot encoded: [0.0, 0.0, ..., 1.0, 0.0, ..., 0.0] (1.0 at index 14)
    - not one-hot encoded: [14.0]
    """

    def __init__(self, tz=timezone.utc, one_hot_encoded: bool = True) -> None:
        super().__init__()
        self.tz = tz
        self.one_hot_encoded = one_hot_encoded

    def calc(self, value: Event) -> FloatArray:
        dt = datetime.astimezone(value.time, self.tz)
        day = dt.day - 1  # day of month is 1-31, we want 0-30
        if not self.one_hot_encoded:
            return np.array([day], dtype=np.float32)

        result = np.zeros(31, dtype=np.float32)
        result[day] = 1.0
        return result

    def size(self) -> int:
        return 31 if self.one_hot_encoded else 1


class MonthOfYearFeature(Feature[Event]):
    """Calculate a month of year where January == 0 and December == 11.
    Result can be one-hot encoded or not, depending on the `one_hot_encoded` parameter.
    If `one_hot_encoded` is True, the result will be a 12-element array else the result will be a single value.
    For example, if the event time is in March, the result will be:
    - one-hot encoded: [0.0, 0.0, 1.0, 0.0, ..., 0.0] (1.0 at index 2)
    - not one-hot encoded: [2.0]
    """

    def __init__(self, tz=timezone.utc, one_hot_encoded: bool = True) -> None:
        super().__init__()
        self.tz = tz
        self.one_hot_encoded = one_hot_encoded

    def calc(self, value: Event) -> FloatArray:
        dt = datetime.astimezone(value.time, self.tz)
        month = dt.month - 1  # month is 1-12, we want 0-11
        if not self.one_hot_encoded:
            return np.array([month], dtype=np.float32)

        result = np.zeros(12, dtype=np.float32)
        result[month] = 1.0
        return result

    def size(self) -> int:
        return 12 if self.one_hot_encoded else 1


class TimeDifference(Feature[Event]):
    """Calculate the time difference in seconds between two consecutive events."""

    def __init__(self) -> None:
        super().__init__()
        self._last_time: datetime | None = None

    def calc(self, value: Event) -> FloatArray:
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
    """Base class for technical analysis features.
    You can for example use TaLib to implement your own technical analysis features.
    """

    def __init__(self, *assets: Asset, period: int) -> None:
        super().__init__()
        self._data: dict[Asset, OHLCVBuffer] = {}
        self.period = period
        self.assets = list(assets)

    def calc(self, value: Event) -> FloatArray:
        result = []
        nan = float("nan")
        for asset in self.assets:
            v = nan
            item = value.price_items.get(asset)
            if isinstance(item, Bar):
                if asset not in self._data:
                    self._data[asset] = OHLCVBuffer(self.period)
                ohlcv = self._data[asset]
                if ohlcv.append(item.ohlcv):
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

        high, low, close = item.ohlcv[1:4]

        prev_close = self.prev_close if self.prev_close is not None else low
        self.prev_close = close

        result = max(high - low, abs(high - prev_close), abs(low - prev_close))

        return np.array([result], dtype=np.float32)

    def size(self) -> int:
        return 1

    def reset(self):
        self.prev_close = None


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
    """Extract the ohlcv values from bars for one or more assets
    So if there are 2 assets, the result will be a 10-element array.
    The order will be: open, high, low, close, volume for each asset.
    """

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, value: Event) -> FloatArray:
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = value.price_items.get(asset)
            if isinstance(item, Bar):
                offset = idx * 5
                result[offset : offset + 5] = item.ohlcv

        return result

    def size(self) -> int:
        return 5 * len(self.assets)


class QuoteFeature(Feature[Event]):
    """Extract the values from quotes for one or more assets.
    So if there are 2 assets, the result will be a 8-element array.
    The order of the values is: ask_price, ask_volume, bid_price, bid_volume,  for each asset.
    """

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, value: Event) -> FloatArray:
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = value.price_items.get(asset)
            if isinstance(item, Quote):
                offset = idx * 4
                result[offset : offset + 4] = item.data

        return result

    def size(self) -> int:
        return 4 * len(self.assets)


class CacheFeature(Feature[Event]):
    """Cache the results of an event feature from a previous run. This can speed up the learning process considerable, but
    this requires that:

    - the feed to have always an increasing time value (monotonic)
    - the feature has to produce the same output at a given time.
    """

    def __init__(self, feature: Feature[Event], validate=False) -> None:
        super().__init__()
        self.feature: Feature = feature
        self._cache: dict[datetime, FloatArray] = {}
        self.validate = validate

    def calc(self, value: Event) -> FloatArray:
        time = value.time
        if time in self._cache:
            values = self._cache[time]
            if self.validate:
                calc_values = self.feature.calc(value)
                assert np.array_equal(values, calc_values, equal_nan=True), (
                    f"Wrong cache time={time} cache={values} calculated={calc_values}"
                )
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
