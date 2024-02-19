import unittest
from decimal import Decimal

from roboquant import Roboquant, Order, PriceItem, FlexTrader, EMACrossover, BasicTracker
from tests.common import get_feed


class _MyTrader(FlexTrader):

    def _get_orders(self, symbol: str, size: Decimal, action: PriceItem, rating: float) -> list[Order]:
        price = action.get_price("CLOSE")
        if price:
            limit_price = price * 0.99 if size > 0 else price * 1.01
            order = Order(symbol, size, limit_price)
            return [order]
        return []


class TestFlexTrader(unittest.TestCase):

    def test_default_flextrader(self):
        feed = get_feed()
        rq = Roboquant(EMACrossover(13, 26), trader=FlexTrader())
        tracker = BasicTracker()
        rq.run(feed, tracker=tracker)
        self.assertGreater(tracker.orders, 0)

    def test_custom_flextrader(self):
        feed = get_feed()
        rq = Roboquant(EMACrossover(13, 26), trader=_MyTrader())
        tracker = BasicTracker()
        rq.run(feed, tracker=tracker)
        self.assertGreater(tracker.orders, 0)


if __name__ == "__main__":
    unittest.main()
