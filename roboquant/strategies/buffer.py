from array import array
from collections import UserDict

import numpy as np
from numpy.typing import NDArray, DTypeLike

from roboquant.asset import Asset
from roboquant.event import Bar, Event


class NumpyBuffer:
    """A FIFO (first-in-first-out) buffer of a fixed capacity.
    It uses a single Numpy array to store its data.
    """

    __slots__ = "__data", "__idx", "rows"

    def __init__(self, rows: int, columns: int, dtype: DTypeLike = "float32", order: str="C") -> None:
        """Create a new Numpy buffer"""
        size = int(rows * 1.25 + 3)  # slight overallocation to minimize copying when buffer is full
        self.__data: NDArray = np.full((size, columns), np.nan, dtype=dtype, order=order)  # type: ignore
        self.__idx = 0
        self.rows = rows

    def append(self, data: array | NDArray | list | tuple) -> bool:
        """Append data to this buffer. Return True if the buffer is full, False otherwise"""
        if self.__idx >= len(self.__data):
            self.__data[0 : self.rows] = self.__data[-self.rows :]
            self.__idx = self.rows

        self.__data[self.__idx] = data
        self.__idx += 1
        return self.__idx >= self.rows

    def __array__(self)-> NDArray:
        start = max(0, self.__idx - self.rows)
        return self.__data[start : self.__idx]

    def _get(self, column: int) -> NDArray:
        start = max(0, self.__idx - self.rows)
        return self.__data[start : self.__idx, column]

    def __len__(self):
        return min(self.__idx, self.rows)

    def is_full(self) -> bool:
        return self.__idx >= self.rows

    def reset(self):
        """reset the buffer"""
        self.__data.fill(np.nan)
        self.__idx = 0


class OHLCVBuffer(NumpyBuffer):
    """A OHLCV buffer (first-in-first-out) of a fixed capacity.
    It stores the data in a `NumpyBuffer`.
    """

    def __init__(self, capacity: int, dtype: DTypeLike = "float64") -> None:
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
    """A OHLCV buffer that tracks multiple assets"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def add_event(self, event: Event) -> set[Asset]:
        """Add a new event and return all the assets that have been added and are ready to be processed.
        PriceItems in the event that are not of the type `Bar` are ignored.
        """
        assets: set[Asset] = set()
        for item in event.items:
            if isinstance(item, Bar):
                asset = item.asset
                if asset not in self:
                    self[asset] = OHLCVBuffer(self.size)
                ohlcv = self[asset]
                if ohlcv.append(item.ohlcv):
                    assets.add(asset)
        return assets

    def ready_assets(self) -> set[Asset]:
        """Return the set of assets for which the buffer is already full"""
        return {asset for asset, ohlcv in self.items() if ohlcv.is_full()}
