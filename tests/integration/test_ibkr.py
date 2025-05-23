import logging
import time
import unittest
from decimal import Decimal

from roboquant import Order
from roboquant.asset import Stock
from roboquant.monetary import Amount, One2OneConversion
from dotenv import load_dotenv

load_dotenv()

from roboquant.ibkr.broker import IBKRBroker  # noqa: E402


class TestIBKR(unittest.TestCase):

    def test_account(self):
        broker = IBKRBroker()
        account = broker.sync()
        print(account)
        self.assertTrue(account.equity_value() > 0)


    def test_ibkr_order(self):
        Amount.register_converter(One2OneConversion())
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("ibapi").setLevel(logging.WARNING)
        asset = Stock("JPM")
        limit = 210.0

        broker = IBKRBroker()

        account = broker.sync()
        self.assertGreater(account.equity_value(), 0)
        old_len_orders = len(account.orders)
        existing_orders = {order.id for order in account.orders}

        # Place an order
        order = Order(asset, 10, limit)
        broker.place_orders([order])
        self.assertEqual(broker.metrics["new"], 1)
        time.sleep(5)
        self.assertEqual(len(account.orders), old_len_orders)
        account = broker.sync()
        print(account)
        self.assertEqual(len(account.orders), old_len_orders + 1)

        new_orders = [order for order in account.orders if order.id not in existing_orders]
        if new_orders:
            new_order = new_orders[-1]
            order_id = new_order.id or -1

            self.assertEqual(new_order.size, Decimal(10))
            self.assertEqual(new_order.limit, limit)
            self.assertEqual(asset, new_order.asset)

            # Update an order
            new_limit = limit - 1.0
            update_order = new_order.modify(5, new_limit)
            broker.place_orders([update_order])
            self.assertEqual(broker.metrics["update"], 1)
            time.sleep(5)

            account = broker.sync()
            print(account)
            new_order = [order for order in account.orders if order.id == order_id][-1]
            print(new_order)
            # self.assertEqual(new_order.size, Decimal(5), new_order)
            # Bug in IBKR
            # self.assertEqual(new_order.limit, new_limit)

            # Cancel an order
            cancel_order = update_order.cancel()
            broker.place_orders([cancel_order])
            self.assertEqual(broker.metrics["cancel"], 1)
            time.sleep(5)
            account = broker.sync()
            cancel_orders = [order for order in account.orders if order.id == order_id]
            if cancel_orders:
                cancel_order = cancel_orders[-1]
                self.assertEqual(cancel_order.size, Decimal())
                self.assertEqual(len(account.orders), old_len_orders + 1)


    def test_cancel_all_orders(self):
        broker = IBKRBroker()
        account = broker.sync()
        print(account)
        cancel_orders = [order.cancel() for order in account.orders if order.size]
        for order in cancel_orders:
            order.info = {"outside_rth" : True}
        broker.place_orders(cancel_orders)
        time.sleep(10)
        account = broker.sync()
        print(account)



if __name__ == "__main__":
    unittest.main()
