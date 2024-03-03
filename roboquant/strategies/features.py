from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from roboquant import Signal
from roboquant.event import Event, Candle
from roboquant.feeds.feed import Feed
from roboquant.strategies.strategy import Strategy


class Feature(ABC):

    @abstractmethod
    def calc(self, evt: Event) -> NDArray:
        """
        Return the result as a 1-dimensional NDArray.
        The result should always be the same size.
        """

    def returns(self, period=1):
        if period == 1:
            return ReturnsFeature(self)
        return LongReturnsFeature(self, period)


class FixedValueFeature(Feature):

    def __init__(self, value: NDArray) -> None:
        super().__init__()
        self.value = value

    def calc(self, evt):
        return self.value


class PriceFeature(Feature):

    def __init__(self, symbol: str, price_type: str = "DEFAULT") -> None:
        self.symbol = symbol
        self.price_type = price_type
        self.name = f"{symbol}-{price_type}-PRICE"

    def calc(self, evt):
        item = evt.price_items.get(self.symbol)
        price = item.price(self.price_type) if item else float("nan")
        return np.array([price])


class CandleFeature(Feature):

    def __init__(self, symbol: str) -> None:

        self.symbol = symbol
        self.name = f"{symbol}-CANDLE"

    def calc(self, evt):
        item = evt.price_items.get(self.symbol)
        if isinstance(item, Candle):
            return np.array(item.ohlcv)

        return np.full((5,), float("nan"))


class VolumeFeature(Feature):

    def __init__(self, symbol, volume_type: str = "DEFAULT") -> None:
        super().__init__()
        self.symbol = symbol
        self.volume_type = volume_type

    def calc(self, evt: Event):
        price_data = evt.price_items.get(self.symbol)
        volume = price_data.volume(self.volume_type) if price_data else float("nan")
        return np.array([volume])


class ReturnsFeature(Feature):

    def __init__(self, feature: Feature) -> None:
        super().__init__()
        self.history = None
        self.feature: Feature = feature

    def calc(self, evt):
        values = self.feature.calc(evt)

        if self.history is None:
            self.history = values
            return np.full(values.shape, float("nan"))

        r = values / self.history - 1.0
        self.history = values
        return r


class LongReturnsFeature(Feature):

    def __init__(self, feature: Feature, period: int) -> None:
        super().__init__()
        self.history = deque(maxlen=period)
        self.feature: Feature = feature

    def calc(self, evt):
        values = self.feature.calc(evt)
        h = self.history

        if len(h) < h.maxlen:  # type: ignore
            h.append(values)
            return np.full(values.shape, float("nan"))

        r = values / h[0] - 1.0
        h.append(values)
        return r


class SMAFeature(Feature):

    def __init__(self, feature, period) -> None:
        super().__init__()
        self.period = period
        self.feature: Feature = feature
        self.history = None
        self._cnt = 0

    def calc(self, evt):
        values = self.feature.calc(evt)
        if self.history is None:
            self.history = np.zeros((self.period, values.size))

        idx = self._cnt % self.period
        self.history[idx] = values
        self._cnt += 1

        if self._cnt < self.period:
            return np.full((values.size,), np.nan)

        return np.mean(self.history, axis=0)


class DayOfWeekFeature(Feature):

    def __init__(self, tz=timezone.utc) -> None:
        self.tz = tz

    def calc(self, evt):
        dt = datetime.astimezone(evt.time, self.tz)
        weekday = dt.weekday()
        result = np.zeros(7)
        result[weekday] = 1.0
        return result


class FeatureStrategy(Strategy, ABC):
    """Abstract base class for strategies based on one or mroe features"""

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
        data = [feature.calc(evt) for feature in features]
        return np.hstack(data, dtype=self._dtype)

    def _get_xy(self, feed: Feed, timeframe=None, warmup=0) -> tuple[NDArray, NDArray]:
        channel = feed.play_background(timeframe)
        x = []
        y = []
        while evt := channel.get():
            if warmup:
                for f in self._features_x:
                    f.calc(evt)
                for f in self._features_y:
                    f.calc(evt)
                warmup -= 1
            else:
                x.append(self.__get_row(evt, self._features_x))
                y.append(self.__get_row(evt, self._features_y))

        return np.asarray(x, dtype=self._dtype), np.asarray(y, dtype=self._dtype)
