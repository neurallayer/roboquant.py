import unittest
from decimal import Decimal

from roboquant import Account, Position, OptionAccount


class TestAccount(unittest.TestCase):

    def test_account_init(self):
        acc = Account()
        self.assertEqual(acc.buying_power, 0.0)
        self.assertEqual(acc.unrealized_pnl({}), 0.0)
        self.assertEqual(acc.mkt_value({}), 0.0)

    def test_account_positions(self):
        acc = Account()
        prices = {}
        for i in range(10):
            symbol = f"AA${i}"
            price = 10.0 + i
            acc.positions[symbol] = Position(Decimal(10), price)
            prices[symbol] = price

        self.assertAlmostEqual(acc.mkt_value(prices), 1450.0)
        self.assertAlmostEqual(acc.unrealized_pnl(prices), 0.0)

    def test_account_option(self):
        acc = OptionAccount()
        acc.register("DUMMY", 5.0)
        self.assertEqual(1000.0, acc.contract_value("DUMMY", Decimal(1), 200.0))
        self.assertEqual(200.0, acc.contract_value("TSLA", Decimal(1), 200.0))

        self.assertEqual(20000.0, acc.contract_value("AAPL  131101C00470000", Decimal(1), 200.0))
        self.assertEqual(2000.0, acc.contract_value("AAPL7 131101C00470000", Decimal(1), 200.0))


if __name__ == "__main__":
    unittest.main()
