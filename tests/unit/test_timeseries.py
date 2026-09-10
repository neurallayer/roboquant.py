from matplotlib.axes import Axes
from pandas import DataFrame
import unittest

from tests.common import get_feed
from roboquant.common.timeseries import TimeSeries


class TestTimeSeries(unittest.TestCase):

    def test_basic(self):

        feed = get_feed()
        symbols = {a.symbol for a in feed.assets()}
        ts = feed.to_timeseries()

        self.assertSetEqual(set(ts.columns), symbols)

        apple = feed.get_asset("AAPL")
        ts = feed.to_timeseries(apple)
        # only works because AAPL has prices from
        # beginning till end in the feed
        self.assertEqual(ts.timeframe(), feed.timeframe())

        self.assertIsInstance(ts, DataFrame)
        ts3 = ts[10:20]
        self.assertEqual(len(ts3), 10)
        self.assertIsInstance(ts3, TimeSeries)

        ax = ts3.plot()
        self.assertIsInstance(ax, Axes)


if __name__ == "__main__":
    unittest.main()
