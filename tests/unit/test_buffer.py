from array import array
import unittest
import numpy as np

from roboquant.strategies.buffer import NumpyBuffer, OHLCVBuffer


class TestBuffer(unittest.TestCase):
    def test_buffer(self):
        b = NumpyBuffer(10, 5)
        x = np.arange(100).reshape(20, 5)
        for row in x:
            b.append(row)

        self.assertRaises(ValueError, b.append, np.arange(6))

        c = np.asarray(b)
        e = c == np.arange(50, 100).reshape(10, 5)
        self.assertTrue(e.all())

    def test_buffer_append(self):
        b = NumpyBuffer(10, 2)
        b.append(array("f", [1, 2]))
        b.append([3, 4])
        b.append((5, 6))
        a = np.asarray(b)
        self.assertEqual(3, len(a))

    def test_ohlcv_buffer(self):
        b = OHLCVBuffer(10)
        for i in range(12):
            b.append([100+i, 101+i, 99+i, 100+i, 5000])
        self.assertTrue(b.is_full())
        self.assertTrue(np.all(b.open() == b.close()))
        c = b.close()
        self.assertEqual(10, len(c))
        self.assertEqual(111, c[-1])  # Check the last close price


if __name__ == "__main__":
    unittest.main()
