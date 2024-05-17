import unittest

from roboquant import Signal, SignalType


class TestSignal(unittest.TestCase):

    def test_signal(self):
        s = Signal.buy("XYZ")
        self.assertEqual(1.0, s.rating)
        self.assertTrue(s.is_buy)
        self.assertFalse(s.is_sell)
        self.assertTrue(s.is_entry)
        self.assertTrue(s.is_exit)
        self.assertTrue(SignalType.ENTRY in s.type)
        self.assertTrue(SignalType.EXIT in s.type)

    def test_signal_equal(self):
        x = Signal("XYZ", -1.0, SignalType.ENTRY_EXIT)
        y = Signal("XYZ", -1.0, SignalType.ENTRY_EXIT)
        self.assertEqual(x, y)

    def test_signal_rating(self):
        s = Signal("XYZ", 0.5, SignalType.ENTRY)
        self.assertEqual(0.5, s.rating)
        self.assertTrue(s.is_entry)
        self.assertFalse(s.is_exit)

    def test_type(self):
        t = SignalType.ENTRY
        self.assertEqual("ENTRY", str(t))
        self.assertEqual("ENTRY_EXIT", str(SignalType.ENTRY_EXIT))


if __name__ == "__main__":
    unittest.main()
