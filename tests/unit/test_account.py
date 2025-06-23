import unittest
from decimal import Decimal

from roboquant.account import Account, Position
from roboquant.asset import Stock
from roboquant.monetary import Wallet, Amount, USD
from roboquant.timeframe import utcnow


class TestAccount(unittest.TestCase):

    def test_account_without(self):
        acc = Account()
        acc.cash = Wallet(Amount(USD, 1_000.0))
        now = utcnow()
        acc.last_update = now
        self.assertEqual(acc.buying_power.value, 0.0)
        self.assertEqual(acc.buying_power.currency, USD)
        self.assertEqual(acc.base_currency, USD)
        self.assertEqual(acc.unrealized_pnl().convert_to(USD, now), 0.0)
        self.assertEqual(acc.realized_pnl().convert_to(USD, now), 0.0)
        self.assertEqual(acc.mkt_value().convert_to(USD, now), 0.0)
        self.assertEqual(acc.equity_value(), 1_000.0)
        self.assertEqual(acc.last_update, now)
        self.assertEqual(len(acc.positions), 0)
        self.assertEqual(len(acc.trades), 0)

    def test_account_with_positions(self):
        acc = Account()
        now = utcnow()
        acc.cash = Wallet(Amount(USD, 1_000.0))
        for i in range(5):
            symbol = Stock(f"AA${i}")
            acc.positions[symbol] = Position(Decimal(10), 10.0, 11.0)

        self.assertAlmostEqual(acc.mkt_value().convert_to(USD, now), 5*10*11.0)
        self.assertAlmostEqual(acc.equity_value(), 1_000.0 + (5*10*11.0))
        self.assertAlmostEqual(acc.unrealized_pnl().convert_to(USD, now), 50.0)


if __name__ == "__main__":
    unittest.main()
