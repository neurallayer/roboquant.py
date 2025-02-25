import unittest
from datetime import timedelta

from roboquant.feeds.util import BarAggregatorFeed, TimeGroupingFeed
from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_price_item_feed


class TestFeedUtil(unittest.TestCase):

    def test_bar_aggregator_feed(self):
        feed = RandomWalk(price_type="trade", frequency=timedelta(seconds=1))
        candle_feed = BarAggregatorFeed(feed, timedelta(seconds=60), price_type="trade")
        run_price_item_feed(candle_feed, feed.assets(), self)

    def test_time_grouping_feed(self):
        feed = RandomWalk(price_type="trade", frequency=timedelta(seconds=1))
        grouped_feed = TimeGroupingFeed(feed, timeout=10.0)
        run_price_item_feed(grouped_feed, feed.assets(), self)


if __name__ == "__main__":
    unittest.main()
