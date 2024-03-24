import pathlib
import unittest

from roboquant.feeds import CSVFeed
from tests.common import run_price_item_feed


class TestCSVFeed(unittest.TestCase):

    @staticmethod
    def _get_root_dir(*paths):
        root = pathlib.Path(__file__).parent.resolve().joinpath("..", "data", *paths)
        return str(root)

    def test_csv_feed_generic(self):
        root = self._get_root_dir("csv")
        feed = CSVFeed(root, time_offset="21:00:00+00:00")
        run_price_item_feed(feed, ["AAPL", "AMZN", "TSLA"], self)

    def test_csv_feed_yahoo(self):
        root = self._get_root_dir("yahoo")
        feed = CSVFeed.yahoo(root)
        run_price_item_feed(feed, ["META"], self)

    def test_csv_feed_stooq_daily(self):
        root = self._get_root_dir("stooq", "daily")
        feed = CSVFeed.stooq_us_daily(root)
        run_price_item_feed(feed, ["IBM"], self)

    def test_csv_feed_stooq_intraday(self):
        root = self._get_root_dir("stooq", "5_min")
        feed = CSVFeed.stooq_us_intraday(root)
        run_price_item_feed(feed, ["IBM"], self)


if __name__ == "__main__":
    unittest.main()
