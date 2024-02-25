import unittest
from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_priceitem_feed


class TestRandomWalk(unittest.TestCase):

    def test_randomwalk(self):
        feed = RandomWalk(n_prices=333, n_symbols=13)
        self.assertEqual(333, len(feed.timeline()))
        self.assertEqual(13, len(feed.symbols))
        run_priceitem_feed(feed, feed.symbols, self)


if __name__ == "__main__":
    unittest.main()
