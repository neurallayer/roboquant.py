import unittest

from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_price_item_feed


class TestRandomWalk(unittest.TestCase):

    def test_randomwalk_bar(self):
        n_prices = 100
        n_symbols = 10
        feed = RandomWalk(n_prices=n_prices, n_symbols=n_symbols, price_type="bar")
        self.assertEqual(n_prices, len(feed.timeline()))
        self.assertEqual(n_symbols, len(feed.assets()))
        run_price_item_feed(feed, feed.assets(), self)

    def test_randomwalk_trade(self):
        feed = RandomWalk(n_prices=50, n_symbols=5, price_type="trade")
        run_price_item_feed(feed, feed.assets(), self)


if __name__ == "__main__":
    unittest.main()
