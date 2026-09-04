import os
import time
import unittest
from decimal import Decimal

from dotenv import load_dotenv

from roboquant.brokers.saxo import SaxoBroker
from roboquant.common.asset import Stock
from roboquant.common.order import Order

load_dotenv()


class TestSaxoBroker(unittest.TestCase):


    def validate(self, broker: SaxoBroker, order: Order) -> Order:
        broker.place_orders([order])
        time.sleep(2)
        account = broker.sync()
        print(account, "\n")
        orders = account.orders_for_asset(order.asset)
        self.assertEqual(len(orders), 1)
        new_order = orders[0]
        self.assertTrue(new_order.id)
        self.assertAlmostEqual(new_order.limit or 0, order.limit or 0)
        self.assertEqual(new_order.tif, order.tif)
        self.assertEqual(new_order.asset, order.asset)
        self.assertEqual(new_order.size, order.size)
        self.assertEqual(new_order.fill, Decimal(0))
        return new_order

    def test_saxo_broker(self):
        key = os.environ["SAXO"]
        broker = SaxoBroker(key)
        account = broker.sync()
        print(account, "\n")
        self.assertTrue(account.buying_power.value > 0)

        apple = Stock("AAPL")
        apple = broker.match_asset(apple)
        assert apple

        # cleanup any pending open apple orders
        cancellations = [order.cancel() for order in account.orders_for_asset(apple)]
        broker.place_orders(cancellations)
        self.assertEqual(broker.metrics["cancel"], len(cancellations))

        # Add an Apple order with a limit that is very unlikely to execute
        # even if the market is open
        price = broker.get_price(apple)
        limit = round(price*0.95,0)
        order = Order(apple, Decimal(11), limit=limit, tif="GTC")
        new_order = self.validate(broker, order)

        # Modify the limit of an order
        order = new_order.modify(limit = round(price*0.96,0))
        new_order = self.validate(broker, order)

        # cleanup the apple order
        cancellations = [new_order.cancel()]
        broker.place_orders(cancellations)

        time.sleep(2)
        account = broker.sync()
        self.assertEqual(account.orders_for_asset(apple), [])
        print(account)

        m = broker.metrics
        self.assertEqual(m["new"], 1)
        self.assertEqual(m["update"], 1)


if __name__ == "__main__":
    unittest.main()
