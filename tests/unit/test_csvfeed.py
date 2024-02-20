import pathlib
import unittest
from roboquant.feeds.csvfeed import CSVFeed
from tests.common import run_priceaction_feed


class TestCSVFeed(unittest.TestCase):

    def test_csvfeed_generic(self):
        root = pathlib.Path(__file__).parent.resolve().joinpath("data", "csv")
        feed = CSVFeed(root, time_offset="21:00:00+00:00")
        run_priceaction_feed(feed, ["AAPL", "AMZN", "TSLA"], self)

    def test_csvfeed_yahoo(self):
        root = pathlib.Path(__file__).parent.resolve().joinpath("data", "yahoo")
        feed = CSVFeed.yahoo(root)
        run_priceaction_feed(feed, ["META"], self)

    def test_csvfeed_stooq(self):
        root = pathlib.Path(__file__).parent.resolve().joinpath("data", "stooq")
        feed = CSVFeed.stooq_us_daily(root)
        run_priceaction_feed(feed, ["IBM"], self)


if __name__ == "__main__":
    unittest.main()
