import unittest
from decimal import Decimal

from roboquant import Order


class TestOrder(unittest.TestCase):

    def test_order_create(self):
        order = Order("AAPL", 100)
        self.assertEqual(Decimal(100), order.size)

        order = Order("AAPL", 100, 120.0)
        self.assertEqual(120.0, order.limit)

        order = Order("AAPL", 100.0)
        self.assertEqual(Decimal(100), order.size)

        order = Order("AAPL", "100.0")
        self.assertEqual(Decimal(100), order.size)

        order = Order("AAPL", Decimal(100))
        self.assertEqual(Decimal(100), order.size)

    def test_order_info(self):
        order = Order("AAPL", 100, tif="ABC")
        info = order.info
        self.assertIn("tif", info)

        order.id = "test"
        update = order.update(size=50)
        info = update.info
        self.assertIn("tif", info)

    def test_order_update(self):
        order = Order("AAPL", 100)
        order.id = "update1"

        update_order = order.update(size=50)
        self.assertEqual(Decimal(100), order.size)
        self.assertEqual(Decimal(50), update_order.size)
        self.assertEqual(order.id, update_order.id)

    def test_order_cancel(self):
        order = Order("AAPL", 100)
        order.id = "cancel1"

        cancel_order = order.cancel()
        self.assertFalse(order.is_cancellation)
        self.assertTrue(cancel_order.is_cancellation)
        self.assertEqual(order.id, cancel_order.id)


if __name__ == "__main__":
    unittest.main()
