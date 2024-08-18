import unittest

from roboquant.asset import Asset, Crypto, Stock
from roboquant.monetary import USD, Currency


class TestAsset(unittest.TestCase):

    def test_stock(self):
        tesla = Stock("TSLA")
        self.assertEqual("TSLA", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        self.assertEqual("Stock", tesla.type())

        v = tesla.serialize()
        testla2 = Asset.deserialize(v)
        self.assertEqual(tesla, testla2)

    def test_crypto(self):
        btc = Crypto.from_symbol("BTC/USDT")
        self.assertEqual("BTC/USDT", btc.symbol)
        self.assertEqual(Currency("USDT"), btc.currency)
        self.assertEqual("Crypto", btc.type())

        v = btc.serialize()
        testla2 = Asset.deserialize(v)
        self.assertEqual(btc, testla2)


if __name__ == "__main__":
    unittest.main()
