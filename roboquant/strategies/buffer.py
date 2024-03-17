from array import array
from typing import Any

import numpy as np
from numpy.typing import NDArray


class NumpyBuffer:
    """A FIFO (first-in-first-out) buffer of a fixed capacity.
    It uses a single Numpy array to store its data.
    """

    __slots__ = "_data", "_idx", "rows"

    def __init__(self, rows: int, columns: int, dtype: Any = "float32", order="C") -> None:
        """Create a new Numpy buffer"""
        size = int(rows * 1.25 + 3)
        self._data: NDArray = np.full((size, columns), np.nan, dtype=dtype, order=order)  # type: ignore
        self._idx = 0
        self.rows = rows

    def append(self, data: array | NDArray | list | tuple):
        if self._idx >= len(self._data):
            self._data[0: self.rows] = self._data[-self.rows:]
            self._idx = self.rows

        self._data[self._idx] = data
        self._idx += 1

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
    """A OHLCV buffer (first-in-first-out) of a fixed capacity."""

    def __init__(self, capacity: int, dtype="float64") -> None:
        """Create a new OHLCV buffer"""
        super().__init__(capacity, 5, dtype)

    def open(self) -> NDArray:
        """Return the open prices"""
        return self._get(0)

    def high(self) -> NDArray:
        """Return the high prices"""
        return self._get(1)

    def low(self) -> NDArray:
        """Return the low prices"""
        return self._get(2)

    def close(self) -> NDArray:
        """Return the close prices"""
        return self._get(3)

    def volume(self) -> NDArray:
        """Return the volumes"""
        return self._get(4)
