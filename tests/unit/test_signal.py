import unittest
from roboquant.signal import Signal, SignalType
from roboquant.asset import Stock


apple = Stock("AAPL")


class TestSignal(unittest.TestCase):

    def test_signal_all(self):
        s1 = Signal(apple, 1.0)
        self.assertEqual(s1.type, SignalType.ENTRY_EXIT)
        self.assertEqual(s1.rating, 1.0)
        self.assertTrue(s1.is_buy)
        self.assertFalse(s1.is_sell)
        self.assertTrue(s1.is_entry)
        self.assertTrue(s1.is_exit)

    def test_signal_exit(self):
        s1 = Signal.sell(apple, SignalType.EXIT)
        self.assertEqual(s1.type, SignalType.EXIT)
        self.assertFalse(s1.is_buy)
        self.assertTrue(s1.is_sell)
        self.assertFalse(s1.is_entry)
        self.assertTrue(s1.is_exit)


if __name__ == "__main__":
    unittest.main()
