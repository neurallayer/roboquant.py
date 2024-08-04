import unittest
from datetime import timedelta

from roboquant.feeds.util import AggregatorFeed
from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_price_item_feed


class TestCandleFeed(unittest.TestCase):

    def test_candle_feed(self):
        feed = RandomWalk(price_type="trade", frequency=timedelta(seconds=1))
        candle_feed = AggregatorFeed(feed, timedelta(seconds=60), price_type="trade")
        run_price_item_feed(candle_feed, feed.assets(), self)


if __name__ == "__main__":
    unittest.main()
