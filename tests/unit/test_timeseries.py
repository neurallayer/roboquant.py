import unittest

from roboquant.timeseries import TimeSeries
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

    def test_concat_getitem(self):
        feed = get_feed()
        tf1, tf2 = feed.timeframe().split(2)
        ts1 = feed.get_prices("AAPL", timeframe=tf1)
        ts2 = feed.get_prices("AAPL", timeframe=tf2)

        ts = TimeSeries.concat(ts1, ts2)
        self.assertEqual(ts.timeframe(), feed.timeframe())

        ts3 = ts[10:20]
        self.assertEqual(len(ts3), 10)

        ts4 = ts[1]
        self.assertEqual(len(ts4), 1)
        self.assertEqual(ts4.data[0], ts.data[1])


if __name__ == "__main__":
    unittest.main()
