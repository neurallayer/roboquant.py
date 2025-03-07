from decimal import Decimal
import unittest

from roboquant.asset import Crypto, Stock, Option
from roboquant.monetary import USD, Currency


class TestAsset(unittest.TestCase):

    def test_stock(self):
        tesla = Stock("TSLA")
        self.assertEqual("TSLA", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        v = tesla.serialize()
        testla2 = Stock.deserialize(v)
        self.assertEqual(tesla, testla2)
        self.assertRaises(AssertionError, Stock.deserialize, "Stock2:GOOG:USD")

        cv = tesla.contract_value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0)

    def test_crypto(self):
        btc = Crypto.from_symbol("BTC/USDT")
        self.assertEqual("BTC/USDT", btc.symbol)
        self.assertEqual(Currency("USDT"), btc.currency)
        v = btc.serialize()
        testla2 = Crypto.deserialize(v)
        self.assertEqual(btc, testla2)
        self.assertRaises(AssertionError, Crypto.deserialize, "Crypto2:BTC/USDT:USDT")

    def test_option(self):
        tesla = Option("TSLA250228C00100000")
        self.assertEqual("TSLA250228C00100000", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        v = tesla.serialize()
        testla2 = Option.deserialize(v)
        self.assertEqual(tesla, testla2)
        self.assertRaises(AssertionError, Option.deserialize, "Option2:TSLA250228C00100000:USD")

        cv = tesla.contract_value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0*100)


if __name__ == "__main__":
    unittest.main()
