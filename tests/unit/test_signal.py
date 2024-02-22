import unittest
from roboquant import Signal, BUY, SELL, SignalType


class TestSignal(unittest.TestCase):

    def test_signal(self):
        s = BUY
        self.assertEqual(1.0, s.rating)
        self.assertTrue(s.is_buy)
        self.assertTrue(SignalType.ENTRY in s.type)
        self.assertTrue(SignalType.EXIT in s.type)

        x = SELL
        y = Signal(-1.0, SignalType.BOTH)
        self.assertEqual(x, y)

        s = Signal(0.5, SignalType.ENTRY)
        self.assertEqual(0.5, s.rating)
        self.assertTrue(SignalType.ENTRY in s.type)
        self.assertFalse(SignalType.EXIT in s.type)


if __name__ == "__main__":
    unittest.main()
