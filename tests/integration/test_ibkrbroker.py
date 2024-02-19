from decimal import Decimal
import logging
import time
import unittest
from roboquant import OrderStatus, Order
from roboquant.brokers.ibkrbroker import IBKRBroker


class TestIBKRBroker(unittest.TestCase):

    def test_ibkrbroker(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("ibapi").setLevel(logging.WARNING)
        symbol = "JPM"

        broker = IBKRBroker()

        account = broker.sync()
        self.assertGreater(account.equity, 0)
        self.assertEqual(len(account.orders), 0)

        # Place an order
        order = Order(symbol, 10, 150.0)
        broker.place_orders(order)
        time.sleep(5)
        self.assertEqual(len(account.orders), 0)
        account = broker.sync()
        self.assertEqual(len(account.orders), 1)
        self.assertEqual(account.orders[0].size, Decimal(10))
        self.assertTrue(account.orders[0].open)
        self.assertEqual(symbol, account.orders[0].symbol)

        # Update an order
        update_order = order.update(size=5, limit=160.0)
        broker.place_orders(update_order)
        time.sleep(5)
        account = broker.sync()
        self.assertEqual(len(account.orders), 1)
        self.assertEqual(account.orders[0].size, Decimal(5))
        self.assertEqual(account.orders[0].limit, 160.0)
        self.assertTrue(account.orders[0].open)

        # Cancel an order
        cancel_order = update_order.cancel()
        broker.place_orders(cancel_order)
        time.sleep(5)
        account = broker.sync()
        self.assertEqual(len(account.orders), 1)
        self.assertTrue(account.orders[0].closed)
        self.assertEqual(OrderStatus.CANCELLED, account.orders[0].status)

        print(account)


if __name__ == "__main__":
    unittest.main()
