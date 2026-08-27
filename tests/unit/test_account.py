import unittest
from decimal import Decimal

from roboquant.common.position import Position
from roboquant.common.account import Account
from roboquant.common.asset import Stock
from roboquant.common.monetary import Wallet, Amount, USD
from roboquant.common.timeframe import utcnow


class TestAccount(unittest.TestCase):

    def test_initial_account(self):
        acc = Account.empty()
        acc.cash = Wallet(Amount(USD, 1_000.0))
        now = utcnow()
        acc.last_update = now
        self.assertEqual(acc.buying_power.value, 0.0)
        self.assertEqual(acc.buying_power.currency, USD)
        self.assertEqual(acc.base_currency, USD)
        self.assertEqual(acc.unrealized_pnl(), Wallet())
        self.assertEqual(acc.realized_pnl(), Wallet())
        self.assertEqual(acc.mkt_value(), Wallet())
        self.assertEqual(acc.equity(), acc.cash)
        self.assertEqual(acc.last_update, now)
        self.assertEqual(len(acc.positions), 0)
        self.assertEqual(len(acc.trades), 0)

    def test_account_with_positions(self):
        acc = Account.empty()
        now = utcnow()
        acc.cash = Wallet(Amount(USD, 1_000.0))
        for i in range(5):
            asset = Stock(f"AA{i}")
            acc.positions.append(Position(asset, Decimal(10), 10.0, 11.0))

        self.assertAlmostEqual(acc.mkt_value().convert_to(USD, now), 5*10*11.0)
        self.assertAlmostEqual(acc.equity_value(), 1_000.0 + (5*10*11.0))
        self.assertAlmostEqual(acc.unrealized_pnl().convert_to(USD, now), 50.0)


if __name__ == "__main__":
    unittest.main()
