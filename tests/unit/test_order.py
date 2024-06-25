from datetime import datetime, timedelta, timezone
import unittest
from decimal import Decimal

from roboquant import Order


class TestOrder(unittest.TestCase):

    gtd = datetime.now(tz=timezone.utc) + timedelta(days=10)

    def test_order_create(self):
        order = Order("AAPL", 100, 120.0, self.gtd)
        self.assertEqual(120.0, order.limit)
        self.assertEqual(None, order.id)

    def test_order_info(self):
        order = Order("AAPL", 100,  120.0, self.gtd, tif="ABC")
        info = order.info
        self.assertIn("tif", info)

        order.id = "test"
        update = order.modify(size=50)
        info = update.info
        self.assertIn("tif", info)

    def test_order_update(self):
        order = Order("AAPL", 100, 120.0, self.gtd)
        order.id = "update1"

        update_order = order.modify(size=50)
        self.assertEqual(Decimal(100), order.size)
        self.assertEqual(Decimal(50), update_order.size)
        self.assertEqual(order.id, update_order.id)

    def test_order_cancel(self):
        order = Order("AAPL", 100, 120.0, self.gtd)
        order.id = "cancel1"

        cancel_order = order.cancel()
        self.assertFalse(order.is_cancellation)
        self.assertTrue(cancel_order.is_cancellation)
        self.assertEqual(order.id, cancel_order.id)


if __name__ == "__main__":
    unittest.main()
