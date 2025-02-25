import logging
import time
import unittest
from decimal import Decimal

from roboquant import Order
from roboquant.asset import Stock
from roboquant.ibkr import IBKRBroker
from roboquant.monetary import Amount, One2OneConversion


class TestIBKR(unittest.TestCase):

    def test_ibkr_order(self):
        Amount.register_converter(One2OneConversion())
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("ibapi").setLevel(logging.WARNING)
        asset = Stock("JPM")
        limit = 250.0

        broker = IBKRBroker()
        try:
            account = broker.sync()
            self.assertGreater(account.equity_value(), 0)
            self.assertEqual(len(account.orders), 0)

            # Place an order
            order = Order(asset, 10, limit)
            broker.place_orders([order])
            time.sleep(5)
            self.assertEqual(len(account.orders), 0)
            account = broker.sync()
            print(account)
            self.assertEqual(len(account.orders), 1)
            self.assertEqual(account.orders[0].size, Decimal(10))
            self.assertEqual(account.orders[0].limit, limit)
            self.assertEqual(asset, account.orders[0].asset)

            # Update an order
            update_order = order.modify(size=5, limit=limit - 1)
            broker.place_orders([update_order])
            time.sleep(5)
            account = broker.sync()
            print(account)
            self.assertEqual(len(account.orders), 1)
            self.assertEqual(account.orders[0].size, Decimal(5))
            self.assertEqual(account.orders[0].limit, limit - 1)

            # Cancel an order
            cancel_order = update_order.cancel()
            broker.place_orders([cancel_order])
            time.sleep(5)
            account = broker.sync()
            print(account)
            self.assertEqual(len(account.orders), 0)
        finally:
            broker.disconnect()


if __name__ == "__main__":
    unittest.main()
