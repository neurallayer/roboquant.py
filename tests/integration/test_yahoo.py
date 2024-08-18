import unittest
from datetime import datetime

from roboquant.asset import Stock
from roboquant.feeds import YahooFeed
from tests.common import run_price_item_feed


class TestYahoo(unittest.TestCase):

    def test_yahoo_feed(self):
        symbols = {"MSFT", "JPM"}
        feed = YahooFeed(*symbols, start_date="2018-01-01", end_date="2020-01-01")
        self.assertEqual(2, len(feed.assets()))

        assets = {Stock(symbol) for symbol in symbols}
        self.assertEqual(assets, set(feed.assets()))

        tf = feed.timeframe()
        assert tf
        self.assertTrue(tf.start == datetime.fromisoformat("2018-01-02T05:00:00+00:00"))
        self.assertTrue(tf.end == datetime.fromisoformat("2019-12-31T05:00:00+00:00"))
        self.assertEqual(503, len(feed.timeline()))
        run_price_item_feed(feed, assets, self)

    def test_yahoo_feed_wrong_symbol(self):
        # expect some error logging due to providing an invalid symbol
        feed = YahooFeed("INVALID_TICKER_NAME", start_date="2010-01-01", end_date="2020-01-01")
        self.assertEqual(0, len(feed.assets()))


if __name__ == "__main__":
    unittest.main()
