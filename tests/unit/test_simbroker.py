import unittest
from datetime import datetime, timezone
from decimal import Decimal

from roboquant import Event, Order, Trade
from roboquant.account import Account
from roboquant.asset import Stock
from roboquant.brokers import SimBroker


class TestSimbroker(unittest.TestCase):

    apple = Stock("AAPL")

    @staticmethod
    def _create_event(price=100.0):
        item = Trade(TestSimbroker.apple, price, 1000)
        event = Event(datetime.now(timezone.utc), [item])
        return event

    def assert_orders(self, account: Account):
        for o in account.orders:
            self.assertTrue(o.id)
            self.assertTrue(o.fill < o.size if o.is_buy else o.fill > o.size)

    def test_simbroker(self):
        broker = SimBroker()
        account = broker.sync()
        self.assertEqual(1_000_000.0, account.buying_power.value)

        order = Order(TestSimbroker.apple, 100, 101.0)
        broker.place_orders([order])
        # No sync yet called
        self.assertEqual(len(account.orders), 0)

        # No event provided, so no processing of open orders
        account = broker.sync()
        self.assertEqual(len(account.orders), 1)
        self.assert_orders(account)

        event = self._create_event()
        account = broker.sync(event)
        self.assertEqual(len(account.orders), 0)
        self.assertEqual(len(account.positions), 1)
        self.assert_orders(account)
        self.assertEqual(Decimal(100), account.positions[TestSimbroker.apple].size)

        # Limit should prevent execution
        order = Order(TestSimbroker.apple, -50, 101.0)
        broker.place_orders([order])
        account = broker.sync(event)
        self.assertEqual(len(account.orders), 1)
        self.assert_orders(account)
        self.assertEqual(len(account.positions), 1)
        self.assertEqual(Decimal(100), account.positions[TestSimbroker.apple].size)

        # Lower the limit so it gets executed
        order = order.modify(limit=99.0)
        broker.place_orders([order])
        account = broker.sync(event)
        self.assertEqual(len(account.orders), 0)
        self.assert_orders(account)
        self.assertEqual(len(account.positions), 1)
        self.assertEqual(Decimal(50), account.positions[TestSimbroker.apple].size)


if __name__ == "__main__":
    unittest.main()
