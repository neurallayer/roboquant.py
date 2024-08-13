import unittest

from roboquant.wallet import NoConversion, One2OneConversion, Wallet, Amount


class TestWallet(unittest.TestCase):

    def _update(self):
        a = Amount("USD", 10)
        a.value = 100  # type: ignore

    def test_wallet_init(self):
        w = Wallet(Amount("USD", 100), Amount("EUR", 50))
        w += Amount("USD", 100)

        self.assertEqual(200, w["USD"])
        self.assertEqual(50, w["EUR"])

        v = w.deepcopy()
        self.assertDictEqual(w, v)

        v += Amount("EUR", 100)
        self.assertNotEqual(w, v)

        z = v + v - v
        self.assertDictEqual(z, v)

        self.assertRaises(Exception, self._update)

    def test_conversion(self):

        Amount.converter = One2OneConversion()
        one_dollar = Amount("USD", 1.0)
        self.assertEqual(1.0, one_dollar.convert("EUR"))

        Amount.converter = NoConversion()
        self.assertRaises(NotImplementedError, lambda: one_dollar.convert("EUR"))


if __name__ == "__main__":
    unittest.main()
