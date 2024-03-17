from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from roboquant.signal import Signal
from roboquant.account import Account
from roboquant.event import Event, Bar
from roboquant.feeds.feed import Feed
from roboquant.strategies.strategy import Strategy


class Feature(ABC):

    @abstractmethod
    def calc(self, evt: Event, account: Account | None) -> NDArray:
        """
        Return the result as a 1-dimensional NDArray.
        The result should always be the same size.
        """

    @abstractmethod
    def size(self) -> int:
        "return the size of this feature"

    def returns(self, period=1):
        if period == 1:
            return ReturnsFeature(self)
        return LongReturnsFeature(self, period)

    def __getitem__(self, *args):
        return SlicedFeature(self, args)

    def reset(self):
        """Reset the state of the feature"""

    def _zeros(self):
        return np.zeros((self.size(),), dtype=np.float32)

    def _full_nan(self):
        return np.full((self.size(),), float("nan"), dtype=np.float32)


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


class FillFeature(Feature):
    """If the feature returns nan, use the last complete values instead"""

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.history = None
        self.feature: Feature = feature

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)

        if np.any(np.isnan(values)):
            if self.history is not None:
                return self.history
            return values

        self.history = values
        return values

    def reset(self):
        self.history = None
        self.feature.reset()

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
        self.history = None
        self.feature: Feature = feature

    def calc(self, evt, account):
        values = self.feature.calc(evt, account)

        if self.history is None:
            self.history = values
            return self._full_nan()

        r = values / self.history - 1.0
        self.history = values
        return r

    def size(self) -> int:
        return self.feature.size()

    def reset(self):
        self.history = None
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


class FeatureStrategy(Strategy, ABC):
    """Abstract base class for strategies wanting to use features
    for their input and target.
    """

    def __init__(self, history: int, dtype="float32"):
        self._features_x = []
        self._features_y = []
        self._hist = deque(maxlen=history)
        self._dtype = dtype

    def add_x(self, feature):
        self._features_x.append(feature)

    def add_y(self, feature):
        self._features_y.append(feature)

    def create_signals(self, event: Event) -> dict[str, Signal]:
        h = self._hist
        row = self.__get_row(event, self._features_x)
        h.append(row)
        if len(h) == h.maxlen:
            x = np.asarray(h, dtype=self._dtype)
            return self.predict(x)
        return {}

    @abstractmethod
    def predict(self, x: NDArray) -> dict[str, Signal]: ...

    def __get_row(self, evt, features) -> NDArray:
        data = [feature.calc(evt, None) for feature in features]
        return np.hstack(data, dtype=self._dtype)

    def _get_xy(self, feed: Feed, timeframe=None, warmup=0) -> tuple[NDArray, NDArray]:
        channel = feed.play_background(timeframe)
        x = []
        y = []
        while evt := channel.get():
            if warmup:
                for f in self._features_x:
                    f.calc(evt, None)
                for f in self._features_y:
                    f.calc(evt, None)
                warmup -= 1
            else:
                x.append(self.__get_row(evt, self._features_x))
                y.append(self.__get_row(evt, self._features_y))

        return np.asarray(x, dtype=self._dtype), np.asarray(y, dtype=self._dtype)
