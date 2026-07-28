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

        diff = ts.pct_change()
        self.assertEqual(len(diff), len(ts) -1)
        self.assertTrue(-1 < diff.data.mean() < 1)


if __name__ == "__main__":
    unittest.main()
