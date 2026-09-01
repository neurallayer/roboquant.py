from dataclasses import dataclass
from decimal import Decimal
from typing import override
import unittest

from roboquant.common.asset import Crypto, Stock, Option, Forex, Asset
from roboquant.common.monetary import USD, Currency

SEP = ""

class TestAsset(unittest.TestCase):

    def test_stock(self):
        tesla = Stock("TSLA")
        self.assertEqual("TSLA", tesla.symbol)
        self.assertEqual(USD, tesla.currency)

        self.assertIn(tesla, Stock.assets())
        self.assertIn(tesla, Asset.assets())
        self.assertNotIn(tesla, Forex.assets())
        v = tesla.serialize()
        tesla2 = Stock.deserialize(v)
        self.assertEqual(tesla, tesla2)
        self.assertRaises(AssertionError, Stock.deserialize, f"Stock2{SEP}GOOG{SEP}USD")

        cv = tesla.value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0)

    def test_crypto(self):
        btc = Crypto.from_symbol("BTC/USDT")
        self.assertRaises(ValueError, Crypto.from_symbol, "ABCD")
        self.assertEqual("BTC/USDT", btc.symbol)
        self.assertEqual(Currency("USDT"), btc.currency)
        v = btc.serialize()
        tesla2 = Crypto.deserialize(v)
        self.assertEqual(btc, tesla2)

        self.assertRaises(AssertionError, Crypto.deserialize, f"Crypto2{SEP}BTC/USDT{SEP}USDT")

    def test_forex(self):
        btc = Forex.from_symbol("EUR/USD")
        btc2 = Forex.from_symbol("EURUSD")
        self.assertEqual(btc.currency, btc2.currency)
        self.assertRaises(ValueError, Forex.from_symbol, "ABCD")

        self.assertEqual("EUR/USD", btc.symbol)
        self.assertEqual(Currency("USD"), btc.currency)
        v = btc.serialize()
        tesla2 = Forex.deserialize(v)
        self.assertEqual(btc, tesla2)
        self.assertRaises(AssertionError, Forex.deserialize, f"Forex2{SEP}EUR/USD{SEP}USD")

    def test_option(self):
        tesla = Option("TSLA250228C00100000")
        self.assertEqual("TSLA250228C00100000", tesla.symbol)
        self.assertEqual(USD, tesla.currency)
        v = tesla.serialize()
        tesla2 = Option.deserialize(v)
        self.assertEqual(tesla, tesla2)
        self.assertRaises(AssertionError, Option.deserialize, f"Option2{SEP}TSLA250228C00100000{SEP}USD")

        cv = tesla.value(Decimal(100), 150.0)
        self.assertEqual(cv, 100*150.0*100)

    def test_custom_asset(self):

        @dataclass(frozen=True)
        class CustomAsset(Asset):

            multiplier: int = 2

            @override
            def value(self, size: Decimal, price: float) -> float:
                return super().value(size, price) * self.multiplier

            @override
            def serialize(self) -> str:
                return f"CustomAsset{SEP}{self.symbol}{SEP}{self.currency}{SEP}{self.multiplier}"

            @classmethod
            @override
            def deserialize(cls, value: str) -> "CustomAsset":
                _, symbol, currency_name, multiplier = value.split(SEP)
                return CustomAsset(symbol, Currency(currency_name), int(multiplier))

        a = CustomAsset("TEST/XYZ", Currency("XYZ"), 4)
        v = a.value(Decimal(100), 150.0)
        self.assertEqual(v, 100 * 150.0 * 4)
        serialized = a.serialize()
        self.assertEqual(a, CustomAsset.deserialize(serialized))

        assets = CustomAsset.assets()
        self.assertListEqual(assets, [a])


if __name__ == "__main__":
    unittest.main()
