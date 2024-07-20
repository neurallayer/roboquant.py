import unittest

from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_price_item_feed


class TestRandomWalk(unittest.TestCase):

    def test_randomwalk_bar(self):
        feed = RandomWalk(n_prices=333, n_symbols=13, price_type="bar")
        self.assertEqual(333, len(feed.timeline()))
        self.assertEqual(13, len(feed.assets()))
        run_price_item_feed(feed, feed.assets(), self)

    def test_randomwalk_trade(self):
        feed = RandomWalk(n_prices=333, n_symbols=13, price_type="trade")
        self.assertEqual(333, len(feed.timeline()))
        self.assertEqual(13, len(feed.assets()))
        run_price_item_feed(feed, feed.assets(), self)


if __name__ == "__main__":
    unittest.main()
