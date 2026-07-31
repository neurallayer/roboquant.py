import unittest

from tests.common import get_feed


class TestTimeSeries(unittest.TestCase):

    def test_basic(self):
        feed = get_feed()
        ts = feed.get_prices("AAPL")

        # only works because AAPL has prices from
        # beginning till end in the feed
        self.assertEqual(ts.timeframe(), feed.timeframe())

        df = ts.to_dataframe()
        self.assertEqual(len(ts), len(df))


    def test_getitem(self):
        feed = get_feed()
        ts = feed.get_prices("AAPL")

        ts3 = ts[10:20]
        self.assertEqual(len(ts3), 10)

        ts4 = ts[1]
        self.assertEqual(len(ts4), 1)

    def test_plot(self):
        feed = get_feed()
        ts = feed.get_prices("AAPL")
        ts.plot()


if __name__ == "__main__":
    unittest.main()
