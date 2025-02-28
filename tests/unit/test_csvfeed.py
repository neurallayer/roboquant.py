import pathlib
import unittest

from roboquant.asset import Stock
from roboquant.feeds import CSVFeed
from tests.common import run_price_item_feed


class TestCSVFeed(unittest.TestCase):

    @staticmethod
    def _get_root_dir(*paths):
        root = pathlib.Path(__file__).parent.resolve().joinpath("..", "data", *paths)
        return str(root)

    def test_csv_feed_yahoo(self):
        root = self._get_root_dir("yahoo")
        feed = CSVFeed.yahoo(root)
        symbols = ["META", "AAPL", "AMZN", "TSLA"]
        assets = [Stock(symbol) for symbol in symbols]
        run_price_item_feed(feed, assets, self)

    def test_basic_feed(self):
        root = self._get_root_dir("yahoo")
        feed = CSVFeed.yahoo(root)
        asset = feed.get_asset("AAPL")
        assert asset
        ohlcv = feed.get_ohlcv(asset)
        self.assertEqual(feed.count_events(), len(ohlcv))

    def test_csv_feed_stooq_daily(self):
        root = self._get_root_dir("stooq", "daily")
        feed = CSVFeed.stooq_us_daily(root)
        run_price_item_feed(feed, [Stock("IBM")], self)

    def test_csv_feed_stooq_intraday(self):
        root = self._get_root_dir("stooq", "5_min")
        feed = CSVFeed.stooq_us_intraday(root)
        run_price_item_feed(feed, [Stock("IBM")], self)


if __name__ == "__main__":
    unittest.main()
