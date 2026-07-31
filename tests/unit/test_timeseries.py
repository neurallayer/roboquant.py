import unittest

from tests.common import get_feed


class TestTimeSeries(unittest.TestCase):

    def test_basic(self):

        feed = get_feed()
        symbols = {a.symbol for a in feed.assets()}
        ts = feed.to_timeseries()

        self.assertSetEqual(set(ts.names()), symbols)

        apple = feed.get_asset("AAPL")
        ts = feed.to_timeseries(apple)
        # only works because AAPL has prices from
        # beginning till end in the feed
        self.assertEqual(ts.timeframe(), feed.timeframe())

        df = ts.to_dataframe()
        self.assertEqual(len(ts), len(df))

        ts3 = ts[10:20]
        self.assertEqual(len(ts3), 10)

        ts4 = ts[1]
        self.assertEqual(len(ts4), 1)

        ts.plot()



if __name__ == "__main__":
    unittest.main()
