from array import array

import numpy as np
from numpy.typing import NDArray


class NumpyBuffer:
    """A FIFO (first-in-first-out) buffer of a fixed capacity.
    It uses a single Numpy array to store its data.
    """

    __slots__ = "_data", "_idx", "capacity"

    def __init__(self, columns: int, capacity: int, dtype="float32") -> None:
        """Create a new Numpy buffer"""
        self._data: NDArray = np.full((capacity, columns), np.nan, dtype=dtype)
        self._idx = 0
        self.capacity = capacity

    def append(self, data: array | NDArray):
        idx = self._idx % self.capacity
        self._data[idx] = data
        self._idx += 1

    def _get(self, column):
        if self._idx < self.capacity:
            return self._data[: self._idx, column]

        idx = self._idx % self.capacity
        return np.concatenate([self._data[idx:, column], self._data[:idx, column]])

    def __len__(self):
        return min(self._idx, self.capacity)

    def get_all(self):
        """Return all the values in the buffer"""
        if self._idx < self.capacity:
            return self._data[: self._idx]

        idx = self._idx % self.capacity
        return np.concatenate([self._data[idx:], self._data[:idx]])

    def is_full(self) -> bool:
        return self._idx >= self.capacity

    def reset(self):
        """reset the buffer"""
        self._data.fill(np.nan)
        self._idx = 0


class OHLCVBuffer(NumpyBuffer):
    """A OHLCV buffer (first-in-first-out) of a fixed capacity.
    """

    def __init__(self, capacity: int, dtype="float64") -> None:
        """Create a new OHLCV buffer"""
        super().__init__(5, capacity, dtype)

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
