import unittest
from datetime import timedelta

from roboquant.feeds.candlefeed import CandleFeed
from roboquant.feeds.randomwalk import RandomWalk
from tests.common import run_priceitem_feed


class TestCandleFeed(unittest.TestCase):

    def test_candle_feed(self):
        feed = RandomWalk(frequency=timedelta(seconds=1))
        candle_feed = CandleFeed(feed, timedelta(seconds=60))
        run_priceitem_feed(candle_feed, feed.symbols, self)


if __name__ == "__main__":
    unittest.main()
