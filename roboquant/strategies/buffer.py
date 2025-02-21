from array import array
from collections import UserDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from roboquant.asset import Asset
from roboquant.event import Bar, Event


class NumpyBuffer:
    """A FIFO (first-in-first-out) buffer of a fixed capacity.
    It uses a single Numpy array to store its data.
    """

    __slots__ = "_data", "_idx", "rows"

    def __init__(self, rows: int, columns: int, dtype: Any = "float32", order="C") -> None:
        """Create a new Numpy buffer"""
        size = int(rows * 1.25 + 3)  # slight overallocation to minimize copying when buffer is full
        self._data: NDArray = np.full((size, columns), np.nan, dtype=dtype, order=order)  # type: ignore
        self._idx = 0
        self.rows = rows

    def append(self, data: array | NDArray | list | tuple) -> bool:
        """Append data to this buffer. Return True if the buffer is full, False otherwise"""
        if self._idx >= len(self._data):
            self._data[0: self.rows] = self._data[-self.rows:]
            self._idx = self.rows

        self._data[self._idx] = data
        self._idx += 1
        return self._idx >= self.rows

    def __array__(self):
        start = max(0, self._idx - self.rows)
        return self._data[start: self._idx]

    def _get(self, column):
        start = max(0, self._idx - self.rows)
        return self._data[start: self._idx, column]

    def __len__(self):
        return min(self._idx, self.rows)

    def is_full(self) -> bool:
        return self._idx >= self.rows

    def reset(self):
        """reset the buffer"""
        self._data.fill(np.nan)
        self._idx = 0


class OHLCVBuffer(NumpyBuffer):
    """A OHLCV buffer (first-in-first-out) of a fixed capacity.
    It stores the data in a `NumpyBuffer`.
    """

    def __init__(self, capacity: int, dtype="float64") -> None:
        """Create a new OHLCV buffer"""
        super().__init__(capacity, 5, dtype)

    def open(self) -> NDArray:
        """Return the open prices as a Numpy array"""
        return self._get(0)

    def high(self) -> NDArray:
        """Return the high prices as a Numpy array"""
        return self._get(1)

    def low(self) -> NDArray:
        """Return the low prices as a Numpy array"""
        return self._get(2)

    def close(self) -> NDArray:
        """Return the close prices as a Numpy array"""
        return self._get(3)

    def volume(self) -> NDArray:
        """Return the volumes as a Numpy array"""
        return self._get(4)


class OHLCVBuffers(UserDict[Asset, OHLCVBuffer]):
    """A OHLCV buffer for multiple assets"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def add_event(self, event: Event) -> set[Asset]:
        """Add a new event and return all the assets that have been added and are ready to be processed.
        PriceItems that are not Bars are ignored.
        """
        assets: set[Asset] = set()
        for item in event.items:
            if isinstance(item, Bar):
                asset = item.asset
                if asset not in self:
                    self[asset] = OHLCVBuffer(self.size)
                ohlcv = self[asset]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    assets.add(asset)
        return assets

    def ready(self) -> set[Asset]:
        """Return the set of assets for which the buffer is already full"""
        return {asset for asset, ohlcv in self.items() if ohlcv.is_full()}
