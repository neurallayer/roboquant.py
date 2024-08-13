import unittest
from decimal import Decimal

from roboquant import Account, Position
from roboquant.asset import Stock
from roboquant.monetary import Wallet, Amount


class TestAccount(unittest.TestCase):

    def test_account_init(self):
        acc = Account("USD")
        acc.cash = Wallet(Amount("USD", 1_000.0))
        self.assertEqual(acc.buying_power.value, 0.0)
        self.assertEqual(acc.buying_power.currency, "USD")
        self.assertEqual(acc.base_currency, "USD")
        self.assertEqual(acc.unrealized_pnl().convert("USD"), 0.0)
        self.assertEqual(acc.mkt_value().convert("USD"), 0.0)
        self.assertEqual(acc.equity_value(), 1_000.0)

    def test_account_positions(self):
        acc = Account()
        acc.cash = Wallet(Amount("USD", 1_000.0))
        prices = {}
        for i in range(10):
            symbol = Stock(f"AA${i}", "USD")
            price = 10.0 + i
            acc.positions[symbol] = Position(Decimal(10), price, price)
            prices[symbol] = price

        self.assertAlmostEqual(acc.mkt_value().convert("USD"), 1450.0)
        self.assertAlmostEqual(acc.equity_value(), 2450.0)
        self.assertAlmostEqual(acc.unrealized_pnl().convert("USD"), 0.0)


if __name__ == "__main__":
    unittest.main()
