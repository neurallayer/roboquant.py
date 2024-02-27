from datetime import datetime, timezone
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from roboquant.event import Event
from roboquant.strategies.buffer import NumpyBuffer


class Feature(Protocol):
    name: str

    def calc(self, evt: Event) -> NDArray:
        """
        Return the result as a 1-dimensional NDArray.
        The result should always be the same size.
        """
        ...


class FixedValueFeature(Feature):

    def __init__(self, name, value: NDArray) -> None:
        self.name = name
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


class VolumeFeature(Feature):

    def __init__(self, symbol, volume_type: str = "DEFAULT") -> None:
        self.symbol = symbol
        self.name = f"{symbol}-VOLUME"
        self.volume_type = volume_type

    def calc(self, evt: Event):
        price_data = evt.price_items.get(self.symbol)
        volume = price_data.volume(self.volume_type) if price_data else float("nan")
        return np.array([volume])


class ReturnsFeature(Feature):

    def __init__(self, feature: Feature) -> None:
        self.history = None
        self.feature: Feature = feature
        self.name = f"{feature.name}-RETURNS"

    def calc(self, evt):
        values = self.feature.calc(evt)

        if not self.history:
            nan = float("nan")
            self.history = values
            return np.full(values.shape, [nan])
        else:
            r = values / self.history - 1.0
            self.history = values
            return r


class SMAFeature(Feature):

    def __init__(self, feature, period) -> None:
        self.period = period
        self.feature: Feature = feature
        self.name = f"{feature.name}-SMA{period}"
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
        else:
            return np.mean(self.history, axis=0)


class DayOfWeekFeature(Feature):

    def __init__(self, tz=timezone.utc) -> None:
        self.tz = tz
        self.name = "DAY-OF-WEEK"

    def calc(self, evt):
        dt = datetime.astimezone(evt.time, self.tz)
        weekday = dt.weekday()
        result = np.zeros(7)
        result[weekday] = 1.0
        return result


class FeatureSet:

    def __init__(self, size=1_000, warmup=0) -> None:
        self.size = size
        self.features: list[Feature] = []
        self.warmup = warmup
        self._buffer: NumpyBuffer = None  # type: ignore

    def add(self, feature: Feature):
        self.features.append(feature)

    def data(self) -> NDArray:
        return self._buffer.get_all()

    def process(self, evt: Event):
        data = [feature.calc(evt) for feature in self.features]
        row = np.hstack(data)

        if self._buffer is None:
            self._buffer = NumpyBuffer(row.size, self.size, "float32")

        if not self.warmup:
            self._buffer.append(row)
        else:
            self.warmup -= 1
