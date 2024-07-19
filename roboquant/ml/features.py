from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.event import Event, Bar, Quote
from roboquant.strategies.buffer import OHLCVBuffer


class Feature(ABC):
    """Features allows to:
    - extract features from an event and/or account.
    - transform other features.
    """

    @abstractmethod
    def calc(self, evt: Event, account: Account | None) -> NDArray:
        """
        Return the result as a 1-dimensional NDArray.
        The result should always be the same size. If a value cannot be calculated at a certain
        tiem, it should return a float NaN.
        """

    @abstractmethod
    def size(self) -> int:
        """return the size of this feature"""

    def _shape(self):
        """return the shape of this feature as a tuple"""
        return (self.size(),)

    def returns(self, period=1):
        if period == 1:
            return ReturnsFeature(self)
        return LongReturnsFeature(self, period)

    def normalize(self, min_period=3):
        return NormalizeFeature(self, min_period)

    def cache(self, validate=False):
        return CacheFeature(self, validate)

    def __getitem__(self, *args):
        return SlicedFeature(self, args)

    def reset(self):
        """Reset the state of the feature"""

    def _zeros(self):
        return np.zeros(self._shape(), dtype=np.float32)

    def _ones(self):
        return np.ones(self._shape(), dtype=np.float32)

    def _full_nan(self):
        return np.full(self._shape(), float("nan"), dtype=np.float32)


class SlicedFeature(Feature):
    """Calculate a slice from another feature"""

    def __init__(self, feature: Feature, args) -> None:
        super().__init__()
        self.args = args
        self.feature = feature
        self._size = len(feature._zeros()[args])

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        return values[self.args]

    def size(self):
        return self._size

    def reset(self):
        return self.feature.reset()


class TrueRangeFeature(Feature):
    """Calculates the true range value for a asset"""

    def __init__(self, asset: Asset) -> None:
        super().__init__()
        self.prev_close = None
        self.asset = asset

    def calc(self, evt, account):
        item = evt.price_items.get(self.asset)
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

    def calc(self, evt, account):
        return self.value


class PriceFeature(Feature):
    """Extract a single price for one or more assets"""

    def __init__(self, *assets: Asset, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.assets = assets
        self.price_type = price_type

    def calc(self, evt, account):
        prices = [evt.get_price(asset, self.price_type) for asset in self.assets]
        return np.array(prices, dtype=np.float32)

    def size(self) -> int:
        return len(self.assets)


class EquityFeature(Feature):
    """Returns the total equity value of the account"""

    def calc(self, evt, account):
        assert account is not None
        equity = account.equity_value()
        return np.asarray([equity], dtype=np.float32)

    def size(self) -> int:
        return 1


class PositionSizeFeature(Feature):
    """Extract the position value for one or more assets as the fraction of the total equity"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, evt, account):
        assert account is not None
        result = self._zeros()
        equity = account.equity_value()
        for idx, asset in enumerate(self.assets):
            position = account.positions.get(asset)
            if position:
                value = asset.contract_amount(position.size, position.mkt_price).convert(account.base_currency, evt.time)
                pos_size = value / equity - 1.0
                result[idx] = pos_size
        return result

    def size(self) -> int:
        return len(self.assets)


class PositionPNLFeature(Feature):
    """Extract the pnl percentage for an open position for one or more assets.
    Returns 0.0 if there is no open position for a asset"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, evt, account):
        assert account is not None
        result = self._zeros()
        for idx, asset in enumerate(self.assets):
            position = account.positions.get(asset)
            if position:
                pnl = position.mkt_price / position.avg_price - 1.0
                result[idx] = pnl
        return result

    def size(self) -> int:
        return len(self.assets)


class BarFeature(Feature):
    """Extract the ohlcv values from bars for one or more assets"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, evt, account):
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = evt.price_items.get(asset)
            if isinstance(item, Bar):
                offset = idx * 5
                result[offset: offset + 5] = item.ohlcv

        return result

    def size(self) -> int:
        return 5 * len(self.assets)


class QuoteFeature(Feature):
    """Extract the values from quotes for one or more assets"""

    def __init__(self, *assets: Asset) -> None:
        super().__init__()
        self.assets = assets

    def calc(self, evt, account):
        result = self._full_nan()
        for idx, asset in enumerate(self.assets):
            item = evt.price_items.get(asset)
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

    def calc(self, evt, account):
        data = [feature.calc(evt, account) for feature in self.features]
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

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        self.__update(values)
        return self.__normalize_values(values)

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.existing_aggregate = (self._zero_int(), self._zeros(), self._zeros())
        self.feature.reset()


class FillFeature(Feature):
    """If a feature contains a nan value, use the last known value instead"""

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = self._full_nan()

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        mask = np.isnan(values)
        values[mask] = self.fill[mask]
        self.fill = np.copy(values)
        return values

    def reset(self):
        self.fill = self._full_nan()
        self.feature.reset()

    def size(self) -> int:
        return self.feature.size()


class FillZeroFeature(Feature):
    """If a feature contains a nan value, use the last known value instead"""

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.fill = self._zeros()

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        mask = np.isnan(values)
        values[mask] = self.fill[mask]
        return values

    def reset(self):
        self.feature.reset()

    def size(self) -> int:
        return self.feature.size()


class CacheFeature(Feature):
    """Cache the results of a feature. This requires the feed to have an always increasing time value
    and the feature to produce the same output at a given time.

    Typically, this doesn't work for features that depend on account values.
    """

    def __init__(self, feature: Feature, validate=False) -> None:
        super().__init__()
        self.feature: Feature = feature
        self._cache: dict[datetime, NDArray] = {}
        self.validate = validate

    def calc(self, evt, account):
        time = evt.time
        if time in self._cache:
            values = self._cache[time]
            if self.validate:
                calc_values = self.feature.calc(evt, account)
                assert np.array_equal(
                    values, calc_values, equal_nan=True
                ), f"Wrong cache time={time} cache={values} calculated={calc_values}"
            return values

        values = self.feature.calc(evt, account)
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


class VolumeFeature(Feature):
    """Extract the volume for one or more assets"""

    def __init__(self, *assets: Asset, volume_type: str = "DEFAULT") -> None:
        super().__init__()
        self.assets = assets
        self.volume_type = volume_type

    def calc(self, evt: Event, account):
        volumes = [evt.get_volume(asset, self.volume_type) for asset in self.assets]
        return np.array(volumes, dtype=np.float32)

    def size(self) -> int:
        return len(self.assets)


class ReturnsFeature(Feature):

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.history = self._full_nan()

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
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

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
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

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        h = self.history

        if len(h) < h.maxlen:  # type: ignore
            h.append(values)
            return self._full_nan()

        r = max(h) / h[0] - 1.0
        h.append(values)
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history.clear()
        self.feature.reset()


class MaxReturnFeature2(Feature):
    """Calculate the maximum return over a certain period."""

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.feature: Feature = feature
        self.history = np.full((period, self.size()), float("nan"), dtype=np.float32)
        self.idx = -1

    def calc(self, evt, account):
        self.idx += 1
        values = self.feature.calc(evt, account)
        h = self.history
        h[self.idx] = values

        hist_len = len(h)
        if self.idx < hist_len:
            return self._full_nan()

        root_idx = self.idx % hist_len + 1
        root_idx = root_idx if root_idx < hist_len else 0
        r = np.max(h) / h[root_idx] - 1.0
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.idx = -1
        self.feature.reset()


class MinReturnFeature(Feature):
    """Calculate the minimum return over a certain period.
    This will only work on features that return a single value.
    """

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
        h = self.history

        if len(h) < h.maxlen:  # type: ignore
            h.append(values)
            return self._full_nan()

        r = min(h) / h[0] - 1.0
        h.append(values)
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history.clear()
        self.feature.reset()


class SMAFeature(Feature):

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.period = period
        self.feature: Feature = feature
        self.history = None
        self._cnt = 0

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)
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


class DayOfWeekFeature(Feature):
    """Calculate a one-hot-encoded day of the week where Monday == 0 and Sunday == 6"""

    def __init__(self, tz=timezone.utc) -> None:
        super().__init__()
        self.tz = tz

    def calc(self, evt, account):
        dt = datetime.astimezone(evt.time, self.tz)
        weekday = dt.weekday()
        result = np.zeros(7, dtype=np.float32)
        result[weekday] = 1.0
        return result

    def size(self) -> int:
        return 7


class TimeDifference(Feature):
    """Calculate the time difference in seconds between two consecutive events."""

    def __init__(self) -> None:
        super().__init__()
        self._last_time: datetime | None = None

    def calc(self, evt, account):
        if self._last_time:
            diff = evt.time - self._last_time
            self._last_time = evt.time
            return np.asarray([diff.total_seconds()], dtype="float32")

        self._last_time = evt.time
        return self._full_nan()

    def size(self) -> int:
        return 1

    def reset(self):
        self._last_time = None


class TaFeature(Feature):
    """Base class for technical analysis features"""

    def __init__(self, *assets: Asset, history_size: int) -> None:
        super().__init__()
        self._data: dict[Asset, OHLCVBuffer] = {}
        self._size = history_size
        self.assets = list(assets)

    def calc(self, evt, account):
        result = []
        nan = float("nan")
        for asset in self.assets:
            value = nan
            item = evt.price_items.get(asset)
            if isinstance(item, Bar):
                if asset not in self._data:
                    self._data[asset] = OHLCVBuffer(self._size)
                ohlcv = self._data[asset]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    value = self._calc(asset, ohlcv)

            result.append(value)
        return np.asarray(result, dtype=np.float32)

    @abstractmethod
    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        """Override this method with technical analysis logic"""
        ...

    def size(self) -> int:
        return len(self.assets)

    def reset(self):
        self._data = {}
