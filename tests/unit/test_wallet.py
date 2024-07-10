import unittest

from roboquant.wallet import Wallet, Amount


class TestWallet(unittest.TestCase):

    def _update(self):
        a = Amount("USD", 10)
        a.value = 100  # type: ignore

    def test_wallet_init(self):
        w = Wallet(Amount("USD", 100), Amount("EUR", 50))
        w += Amount("USD", 100)

        self.assertEqual(200, w["USD"])
        self.assertEqual(50, w["EUR"])

        v = w.copy()
        self.assertDictEqual(w, v)

        v += Amount("EUR", 100)
        self.assertNotEqual(w, v)

        z = v + v - v
        self.assertDictEqual(z, v)

        self.assertRaises(Exception, self._update)


if __name__ == "__main__":
    unittest.main()
