from datetime import datetime, timedelta, timezone
import unittest
from decimal import Decimal

from roboquant import Order
from roboquant.asset import Stock


apple = Stock("AAPL")


class TestOrder(unittest.TestCase):

    def test_order_create(self):
        order = Order(apple, 100, 120.0)
        self.assertEqual(120.0, order.limit)
        self.assertEqual(None, order.id)
        self.assertEqual(apple, order.asset)
        self.assertEqual(None, order.gtd)

    def test_order_gtd(self):
        now = datetime.now(timezone.utc)

        order = Order(apple, 100, 120.0)
        self.assertFalse(order.is_expired(now))

        order = Order(apple, 100, 120.0, now + timedelta(days=1))
        self.assertFalse(order.is_expired(now))
        self.assertTrue(order.is_expired(now + timedelta(days=2)))

    def test_order_info(self):
        order = Order(apple, 100, 120.0, tif="ABC")
        info = order.info
        self.assertIn("tif", info)

        order.id = "test"
        update = order.modify(size=50)
        info = update.info
        self.assertIn("tif", info)

    def test_order_update(self):
        order = Order(apple, 100, 120.0)
        order.id = "update1"

        update_order = order.modify(size=50)
        self.assertEqual(Decimal(100), order.size)
        self.assertEqual(Decimal(50), update_order.size)
        self.assertEqual(order.id, update_order.id)

    def test_order_cancel(self):
        order = Order(apple, 100, 120.0)
        order.id = "cancel1"

        cancel_order = order.cancel()
        self.assertFalse(order.is_cancellation)
        self.assertTrue(cancel_order.is_cancellation)
        self.assertEqual(order.id, cancel_order.id)


if __name__ == "__main__":
    unittest.main()
