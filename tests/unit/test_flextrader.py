import unittest

import roboquant as rq
from roboquant.strategies import EMACrossover
from roboquant.journals import BasicJournal
from roboquant.traders import FlexTrader
from tests.common import get_feed


class _MyTrader(FlexTrader):

    def _get_orders(self, symbol, size, action, rating):
        price = action.price("CLOSE")
        if price:
            limit_price = price * 0.99 if size > 0 else price * 1.01
            order = rq.Order(symbol, size, limit_price)
            return [order]
        return []


class TestFlexTrader(unittest.TestCase):

    def test_default_flex_trader(self):
        feed = get_feed()
        journal = BasicJournal()
        rq.run(feed, EMACrossover(), journal=journal)
        self.assertGreater(journal.orders, 0)

    def test_custom_flex_trader(self):
        feed = get_feed()
        journal = BasicJournal()
        rq.run(feed, EMACrossover(), trader=_MyTrader(), journal=journal)
        self.assertGreater(journal.orders, 0)


if __name__ == "__main__":
    unittest.main()
