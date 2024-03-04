import unittest
from datetime import datetime, timezone
from decimal import Decimal

from roboquant import Event, Order, Trade
from roboquant.brokers import SimBroker


class TestSimbroker(unittest.TestCase):

    @staticmethod
    def _create_event(price=100.0):
        item = Trade("AAPL", price, 1000)
        event = Event(datetime.now(timezone.utc), [item])
        return event

    def test_simbroker(self):
        broker = SimBroker(clean_up_orders=False)
        account = broker.sync()
        self.assertEqual(1_000_000.0, account.buying_power)

        order = Order("AAPL", 100)
        broker.place_orders([order])
        self.assertEqual(len(account.orders), 0)

        account = broker.sync()
        self.assertEqual(len(account.orders), 1)
        self.assertEqual(len(account.open_orders()), 1)
        order = account.orders[0]
        self.assertTrue(order.id is not None)
        self.assertTrue(order.open)
        self.assertEqual(Decimal(0), order.fill)

        event = self._create_event()
        account = broker.sync(event)
        self.assertEqual(len(account.orders), 1)
        self.assertEqual(len(account.open_orders()), 0)
        self.assertEqual(len(account.positions), 1)
        order = account.orders[0]
        self.assertTrue(order.id is not None)
        self.assertTrue(order.closed)
        self.assertEqual(order.size, order.fill)
        self.assertEqual(Decimal(100), account.positions["AAPL"].size)

        order = Order("AAPL", -50)
        broker.place_orders([order])
        account = broker.sync(event)
        self.assertEqual(len(account.orders), 2)
        self.assertEqual(len(account.open_orders()), 0)
        self.assertEqual(len(account.positions), 1)
        self.assertEqual(Decimal(50), account.positions["AAPL"].size)

    def test_simbroker_safeguards(self):
        broker = SimBroker()
        order = Order("AAPL", -50)
        order.id = "NON_EXISTING"
        with self.assertRaises(AssertionError):
            broker.place_orders([order])


if __name__ == "__main__":
    unittest.main()
