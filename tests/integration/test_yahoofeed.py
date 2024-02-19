import unittest
from datetime import datetime
from roboquant import YahooFeed
from tests.common import test_priceaction_feed


class TestYahooFeed(unittest.TestCase):

    def test_yahoofeed(self):
        feed = YahooFeed("MSFT", "JPM", start_date="2018-01-01", end_date="2020-01-01")
        self.assertEqual(2, len(feed.symbols))
        self.assertEqual({"MSFT", "JPM"}, set(feed.symbols))
        self.assertTrue(feed.timeframe().start == datetime.fromisoformat("2018-01-02T05:00:00+00:00"))
        self.assertTrue(feed.timeframe().end == datetime.fromisoformat("2019-12-31T05:00:00+00:00"))
        self.assertEqual(503, len(feed.timeline()))

        test_priceaction_feed(feed, ["MSFT", "JPM"], self)

    def test_yahoofeed_wrong_symbol(self):
        # expect some error logging due to parsing an invalid symbol
        feed = YahooFeed("INVALID_TICKER_NAME", start_date="2010-01-01", end_date="2020-01-01")
        self.assertEqual(0, len(feed.symbols))


if __name__ == "__main__":
    unittest.main()
