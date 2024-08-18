from datetime import datetime
import unittest

from roboquant.monetary import NoConversion, One2OneConversion, Wallet, Amount, USD, EUR, StaticConversion, GBP, JPY


class TestMonetary(unittest.TestCase):

    def _update(self):
        a = Amount(USD, 10)
        a.value = 100  # type: ignore

    def test_currency(self):
        w = 12@USD + 20@EUR
        self.assertIn(USD, w)
        self.assertIn(USD, w)
        self.assertIn(EUR, w)
        self.assertNotIn(JPY, w)

    def test_wallet_init(self):
        w = Wallet(Amount(USD, 100), Amount(EUR, 50))
        w += Amount(USD, 100)

        self.assertEqual(200, w[USD])
        self.assertEqual(50, w[EUR])

        v = w.deepcopy()
        self.assertDictEqual(w, v)

        v += Amount(EUR, 100)
        self.assertNotEqual(w, v)

        z = v + v - v
        self.assertDictEqual(z, v)

        self.assertRaises(Exception, self._update)

    def test_conversion(self):
        now = datetime.now()
        Amount.register_converter(One2OneConversion())
        one_dollar = Amount(USD, 1.0)
        self.assertEqual(1.0, one_dollar.convert(EUR, now))

        Amount.register_converter(NoConversion())
        self.assertRaises(NotImplementedError, lambda: one_dollar.convert(EUR, now))

    def test_static_conversion(self):
        now = datetime.now()
        converter = StaticConversion(USD, {EUR: 1.1, GBP: 1.3})
        Amount.register_converter(converter)
        amt1 = Amount(GBP, 100.0)
        self.assertAlmostEqual(118.18181818181817, amt1.convert(EUR, now))

        Amount.register_converter(NoConversion())


if __name__ == "__main__":
    unittest.main()
