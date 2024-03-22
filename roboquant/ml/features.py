from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from roboquant.account import Account
from roboquant.event import Event, Bar


class Feature(ABC):
    """Features allows to create features from an event or account object.
    Many features also allow post transformation after that, like normalization.
    """

    @abstractmethod
    def calc(self, evt: Event, account: Account | None) -> NDArray:
        """
        Return the result as a 1-dimensional NDArray.
        The result should always be the same size.
        """

    @abstractmethod
    def size(self) -> int:
        "return the size of this feature"

    def shape(self):
        return (self.size(),)

    def returns(self, period=1):
        if period == 1:
            return ReturnsFeature(self)
        return LongReturnsFeature(self, period)

    def normalize(self, min_period=3):
        return NormalizeFeature(self, min_period)

    def cache(self):
        return CacheFeature(self)

    def __getitem__(self, *args):
        return SlicedFeature(self, args)

    def reset(self):
        """Reset the state of the feature"""

    def _zeros(self):
        return np.zeros(self.shape(), dtype=np.float32)

    def _ones(self):
        return np.ones(self.shape(), dtype=np.float32)

    def _full_nan(self):
        return np.full(self.shape(), float("nan"), dtype=np.float32)


class SlicedFeature(Feature):

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


class TrueRangeFeature(Feature):
    """Calculates the true range value for a symbol"""

    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.prev_close = None
        self.symbol = symbol

    def calc(self, evt, account):
        item = evt.price_items.get(self.symbol)
        if item is None or not isinstance(item, Bar):
            return np.array([float("nan")])

        ohlcv = item.ohlcv
        high = ohlcv[1]
        low = ohlcv[2]
        close = ohlcv[3]

        prev_close = self.prev_close if self.prev_close is not None else low
        self.prev_close = close

        result = max(high - low, abs(high - prev_close), abs(low - prev_close))

        return np.array([result])

    def size(self) -> int:
        return 1

    def reset(self):
        self.prev_close = None


class FixedValueFeature(Feature):

    def __init__(self, value: NDArray) -> None:
        super().__init__()
        self.value = value

    def size(self) -> int:
        return len(self.value)

    def calc(self, evt, account):
        return self.value


class PriceFeature(Feature):
    """Extract a single price for one or more symbols"""

    def __init__(self, *symbols: str, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.symbols = symbols
        self.price_type = price_type

    def calc(self, evt, account):
        prices = [evt.get_price(symbol, self.price_type) for symbol in self.symbols]
        return np.array(prices, dtype=np.float32)

    def size(self) -> int:
        return len(self.symbols)


class EquityFeature(Feature):
    """Returns the total equity value of the account"""

    def calc(self, evt, account):
        assert account is not None
        equity = account.equity()
        return np.asarray([equity], dtype=np.float32)

    def size(self) -> int:
        return 1


class PositionSizeFeature(Feature):
    """Extract the position value for a symbol as fraction of the total equity"""

    def __init__(self, *symbols: str) -> None:
        super().__init__()
        self.symbols = symbols

    def calc(self, evt, account):
        assert account is not None
        result = self._zeros()
        for idx, symbol in enumerate(self.symbols):
            position = account.positions.get(symbol)
            if position:
                value = account.contract_value(symbol, position.size, position.mkt_price)
                pos_size = value / account.equity() - 1.0
                result[idx] = pos_size
        return result

    def size(self) -> int:
        return len(self.symbols)


class PositionPNLFeature(Feature):
    """Extract the pnl for an open position for a symbol.
    Returns 0.0 if there is no open position for a symbol"""

    def __init__(self, *symbols: str) -> None:
        super().__init__()
        self.symbols = symbols

    def calc(self, evt, account):
        assert account is not None
        result = self._zeros()
        for idx, symbol in enumerate(self.symbols):
            position = account.positions.get(symbol)
            if position:
                pnl = position.mkt_price / position.avg_price - 1.0
                result[idx] = pnl
        return result

    def size(self) -> int:
        return len(self.symbols)


class BarFeature(Feature):
    """Extract the ohlcv values from bars for one or more symbols"""

    def __init__(self, *symbols: str) -> None:
        super().__init__()
        self.symbols = symbols

    def calc(self, evt, account):
        result = self._full_nan()
        for idx, symbol in enumerate(self.symbols):
            item = evt.price_items.get(symbol)
            if isinstance(item, Bar):
                offset = idx * 5
                result[offset: offset + 5] = item.ohlcv

        return result

    def size(self) -> int:
        return 5 * len(self.symbols)


class CombinedFeature(Feature):
    """Combine multiple features into one single feature"""

    def __init__(self, *features: Feature) -> None:
        super().__init__()
        self.features = features
        self._size = sum(feature.size() for feature in self.features)

    def calc(self, evt, account):
        data = [feature.calc(evt, account) for feature in self.features]
        return np.hstack(data, dtype=np.float32)

    def size(self) -> int:
        return self._size


class RunningStats:

    def __init__(self, size) -> None:
        self.existing_aggregate = (0, self._zeros(size), self._zeros(size))

    @staticmethod
    def _zeros(size):
        return np.zeros((size,), dtype=np.float32)

    def push(self, new_value):
        (count, mean, m2) = self.existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        m2 += delta * delta2
        self.existing_aggregate = (count, mean, m2)

    # Retrieve the mean, variance and sample variance from an aggregate
    def get_normalize_values(self):
        (count, mean, m2) = self.existing_aggregate
        if count < 1:
            raise ValueError("not enough data")

        variance = m2 / count
        return mean, np.sqrt(variance)


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
        mask = ~ np.isnan(new_value)
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


class FillFeature(Feature):
    """If a feature returns a nan value, use the last known value instead"""

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


class CacheFeature(Feature):
    """Cache the results of a feature. This requires the feed to have always increasing time value
    and the feature to produce same output.

    Typically this doesn't work for features that depend on account values.
    """

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.feature: Feature = feature
        self._cache: dict[datetime, NDArray] = {}

    def calc(self, evt, account):
        time = evt.time
        if time in self._cache:
            return self._cache[time]

        values = self.feature.calc(evt, account)
        self._cache[time] = values
        return values

    def reset(self):
        """Reset the underlying feature. This doesn't cleear the cache"""
        self.feature.reset()

    def clear(self):
        """Clear all of the cache"""
        self._cache = {}

    def size(self) -> int:
        return self.feature.size()


class VolumeFeature(Feature):
    """Extract the volume for one or more symbols"""

    def __init__(self, *symbols: str, volume_type: str = "DEFAULT") -> None:
        super().__init__()
        self.symbols = symbols
        self.volume_type = volume_type

    def calc(self, evt: Event, account: Account):
        volumes = [evt.get_volume(symbol, self.volume_type) for symbol in self.symbols]
        return np.array(volumes, dtype=np.float32)

    def size(self) -> int:
        return len(self.symbols)


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


class DayOfWeekFeature(Feature):
    """Calculate a one-hot-encoded day of the week, Monday being 0"""

    def __init__(self, tz=timezone.utc) -> None:
        self.tz = tz

    def calc(self, evt, account):
        dt = datetime.astimezone(evt.time, self.tz)
        weekday = dt.weekday()
        result = np.zeros(7)
        result[weekday] = 1.0
        return result

    def size(self) -> int:
        return 7
