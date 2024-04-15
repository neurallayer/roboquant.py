import unittest
from datetime import datetime

from roboquant.feeds import YahooFeed
from tests.common import run_price_item_feed


class TestYahoo(unittest.TestCase):

    def test_yahoo_feed(self):
        feed = YahooFeed("MSFT", "JPM", start_date="2018-01-01", end_date="2020-01-01")
        self.assertEqual(2, len(feed.symbols))
        self.assertEqual({"MSFT", "JPM"}, set(feed.symbols))

        tf = feed.timeframe()
        assert tf
        self.assertTrue(tf.start == datetime.fromisoformat("2018-01-02T05:00:00+00:00"))
        self.assertTrue(tf.end == datetime.fromisoformat("2019-12-31T05:00:00+00:00"))
        self.assertEqual(503, len(feed.timeline()))

        run_price_item_feed(feed, ["MSFT", "JPM"], self)

    def test_yahoo_feed_wrong_symbol(self):
        # expect some error logging due to parsing an invalid symbol
        feed = YahooFeed("INVALID_TICKER_NAME", start_date="2010-01-01", end_date="2020-01-01")
        self.assertEqual(0, len(feed.symbols))


if __name__ == "__main__":
    unittest.main()
