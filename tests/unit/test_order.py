import unittest
from decimal import Decimal

from roboquant.order import Order
from roboquant.asset import Stock


apple = Stock("AAPL")


class TestOrder(unittest.TestCase):

    def test_order_create(self):
        order = Order(apple, 100, 120.0)
        self.assertEqual(120.0, order.limit)
        self.assertEqual("", order.id)
        self.assertEqual(apple, order.asset)
        self.assertEqual("DAY", order.tif)

    def test_order_info(self):
        order = Order(apple, 100, 120.0, extra="ABC")
        info = order.info
        self.assertIn("extra", info)

        order.id = "test"
        update = order.modify(size=50)
        info = update.info
        self.assertIn("extra", info)

    def test_order_update(self):
        order = Order(apple, 100, 120.0)
        self.assertRaises(Exception, order.modify)

        order.id = "update1"
        update_order = order.modify(size=50)
        self.assertEqual(Decimal(100), order.size)
        self.assertEqual(Decimal(50), update_order.size)
        self.assertEqual(order.id, update_order.id)

    def test_order_cancel(self):
        order = Order(apple, 100, 120.0)
        self.assertRaises(Exception, order.cancel)

        order.id = "cancel1"
        cancel_order = order.cancel()
        self.assertFalse(cancel_order.size)
        self.assertEqual(order.id, cancel_order.id)


if __name__ == "__main__":
    unittest.main()
