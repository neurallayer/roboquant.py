import unittest
from roboquant import Signal, BUY, SELL, SignalType


class TestSignal(unittest.TestCase):

    def test_signal(self):
        s = BUY
        self.assertEqual(1.0, s.rating)
        self.assertTrue(s.is_buy)
        self.assertFalse(s.is_sell)
        self.assertTrue(s.is_entry)
        self.assertTrue(s.is_exit)
        self.assertTrue(SignalType.ENTRY in s.type)
        self.assertTrue(SignalType.EXIT in s.type)

    def test_signal_equal(self):
        x = SELL
        y = Signal(-1.0, SignalType.BOTH)
        self.assertEqual(x, y)
        self.assertEqual(BUY, BUY)
        self.assertNotEqual(BUY, SELL)

    def test_signal_rating(self):
        s = Signal(0.5, SignalType.ENTRY)
        self.assertEqual(0.5, s.rating)
        self.assertTrue(s.is_entry)
        self.assertFalse(s.is_exit)


if __name__ == "__main__":
    unittest.main()
