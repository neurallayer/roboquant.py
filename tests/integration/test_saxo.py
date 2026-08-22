from decimal import Decimal
import time
import unittest
import os

from roboquant.common.asset import Stock
from roboquant.common.order import Order
from roboquant.brokers.saxo import SaxoBroker
from dotenv import load_dotenv

load_dotenv()

def _get_credentials():
    return os.environ["SAXO"]

class TestSaxoBroker(unittest.TestCase):


    def test_saxo_broker(self):
        broker = SaxoBroker(_get_credentials())
        broker._load_all_assets()
        account = broker.sync()
        print(account, "\n")
        self.assertTrue(account.buying_power.value > 0)

        # Add apple order
        apple = Stock("AAPL")
        apple = broker.match_asset(apple)
        assert apple
        price = broker.get_price(apple)
        limit = round(price*0.95,0)
        order = Order(apple, Decimal(11), limit=limit)
        broker.place_orders([order])
        time.sleep(2)
        account = broker.sync()
        print(account, "\n")

        # Cancel all apple orders
        for order in account.orders:
            if order.asset == apple:
                self.assertAlmostEqual(order.limit or 0, limit)
                cancel = order.cancel()
                broker.place_orders([cancel])

        time.sleep(2)
        account = broker.sync()
        print(account)


if __name__ == "__main__":
    unittest.main()
