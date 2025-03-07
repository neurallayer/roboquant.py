from datetime import datetime
import unittest

from roboquant.monetary import (
    NoConversion,
    One2OneConversion,
    Wallet,
    Amount,
    USD,
    EUR,
    StaticConversion,
    GBP,
    JPY,
    ECBConversion,
)


class TestMonetary(unittest.TestCase):

    def _update(self):
        a = Amount(USD, 10)
        a.value = 100  # type: ignore

    def test_amount(self):
        amt1 = 100@USD
        self.assertEqual(amt1.value, 100)
        self.assertEqual(amt1.currency, USD)
        amt2 = 200@USD
        w = amt1 + amt2
        self.assertIsInstance(w, Wallet)
        self.assertEqual(w[USD], 300)

    def test_currency(self):
        w = 12 @ USD + 20 @ EUR
        assert isinstance(w, Wallet)
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
        self.assertEqual(1.0, one_dollar.convert_to(EUR, now))

        Amount.register_converter(NoConversion())
        self.assertRaises(NotImplementedError, lambda: one_dollar.convert_to(EUR, now))

    def test_static_conversion(self):
        now = datetime.now()
        converter = StaticConversion(USD, {EUR: 0.9, GBP: 0.8, JPY: 150})
        Amount.register_converter(converter)
        amt1 = Amount(GBP, 100.0)
        self.assertAlmostEqual(112.5, amt1.convert_to(EUR, now))

        start = 100 @ EUR
        self.assertAlmostEqual(start.value, (100 @ EUR @ USD @ EUR).value)

        Amount.register_converter(NoConversion())

    def test_ecb_conversion(self):
        now = datetime.fromisoformat("2020-01-01T00:00:00+00:00")
        ECBConversion(force_download=False).register()
        amt1 = Amount(GBP, 100.0)
        self.assertAlmostEqual(117.8856, amt1.convert_to(EUR, now), 4)

        # convert an amount to its own currency
        self.assertEqual(amt1.value, (amt1 @ amt1.currency).value)
        NoConversion().register()


if __name__ == "__main__":
    unittest.main()
