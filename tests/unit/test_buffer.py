from array import array
import unittest
import numpy as np

from roboquant.strategies.buffer import NumpyBuffer


class TestBuffer(unittest.TestCase):
    def test_buffer(self):
        b = NumpyBuffer(10, 5)
        x = np.arange(100).reshape(20, 5)
        for row in x:
            b.append(row)

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


if __name__ == "__main__":
    unittest.main()
