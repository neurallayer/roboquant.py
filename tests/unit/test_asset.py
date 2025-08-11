from decimal import Decimal
import unittest

from roboquant.asset import Crypto, Stock, Option, Forex, Asset
from roboquant.monetary import USD, Currency


class TestAsset(unittest.TestCase):

    def test_stock(self):
        tesla = Stock("TSLA")
        self.assertEqual("TSLA", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        v = tesla.serialize()
        tesla2 = Stock.deserialize(v)
        self.assertEqual(tesla, tesla2)
        self.assertRaises(AssertionError, Stock.deserialize, "Stock2:GOOG:USD")

        cv = tesla.contract_value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0)

    def test_crypto(self):
        btc = Crypto.from_symbol("BTC/USDT")
        self.assertEqual("BTC/USDT", btc.symbol)
        self.assertEqual(Currency("USDT"), btc.currency)
        v = btc.serialize()
        tesla2 = Crypto.deserialize(v)
        self.assertEqual(btc, tesla2)
        self.assertRaises(AssertionError, Crypto.deserialize, "Crypto2:BTC/USDT:USDT")

    def test_forex(self):
        btc = Forex.from_symbol("EUR/USD")
        self.assertEqual("EUR/USD", btc.symbol)
        self.assertEqual(Currency("USD"), btc.currency)
        v = btc.serialize()
        tesla2 = Forex.deserialize(v)
        self.assertEqual(btc, tesla2)
        self.assertRaises(AssertionError, Forex.deserialize, "Forex2:EUR/USD:USD")

    def test_option(self):
        tesla = Option("TSLA250228C00100000")
        self.assertEqual("TSLA250228C00100000", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        v = tesla.serialize()
        tesla2 = Option.deserialize(v)
        self.assertEqual(tesla, tesla2)
        self.assertRaises(AssertionError, Option.deserialize, "Option2:TSLA250228C00100000:USD")

        cv = tesla.contract_value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0*100)

    def test_custom_asset(self):
        class CustomAsset(Asset):

            def contract_value(self, size: Decimal, price: float) -> float:
                return super().contract_value(size, price) * 1.5

            def serialize(self) -> str:
                return f"CustomAsset:{self.symbol}:{self.currency}"

            @staticmethod
            def deserialize(value: str) -> "CustomAsset":
                _, symbol, currency = value.split(":")
                return CustomAsset(symbol, Currency(currency))

        a = CustomAsset("TEST/XYZ", Currency("XYZ"))
        v = a.contract_value(Decimal(100), 150.0)
        self.assertEqual(v, 100 * 150.0 * 1.5)
        serialized = a.serialize()
        self.assertEqual(a, CustomAsset.deserialize(serialized))


if __name__ == "__main__":
    unittest.main()
