import unittest
from decimal import Decimal
from dataclasses import replace

from roboquant.order import Order
from roboquant.asset import Stock


apple = Stock("AAPL")


class TestOrder(unittest.TestCase):

    def test_order_create(self):
        order = Order(apple, Decimal(100), 120.0)
        self.assertEqual(120.0, order.limit)
        self.assertEqual("", order.id)
        self.assertEqual(apple, order.asset)
        self.assertEqual("DAY", order.tif)

    def test_order_cancel(self):
        order = Order(apple, Decimal(100), 120.0)
        self.assertRaises(Exception, order.cancel)
        order = replace(order, id = "test")
        cancel_order = order.cancel()
        self.assertFalse(cancel_order.size)
        self.assertEqual(order.id, cancel_order.id)

    def test_order_execute(self):
        order = Order(apple, Decimal(100), 100.0)
        self.assertTrue(order.is_executable(99))
        self.assertFalse(order.is_executable(101))


if __name__ == "__main__":
    unittest.main()
