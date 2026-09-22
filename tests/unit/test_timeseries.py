from datetime import timedelta
from typing import Any

from matplotlib.axes import Axes
from pandas import DataFrame
import unittest

from roboquant.common.timeframe import Timeframe
from tests.common import get_feed
from roboquant.common.timeseries import TimeSeries, Timeline


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

    def test_plot(self):
        feed = get_feed()
        asset = feed.get_asset("AAPL")
        ax = feed.plot(asset)
        dt: Any = ax.lines[0].get_xdata()
        for a,b in zip(feed.timeline(), dt):
            self.assertEqual(a,b)

    def test_timeline(self):
        tf = Timeframe.fromisoformat("2020-01-01", "2020-06-01")
        tl = Timeline.from_timeframe(tf, "1 day")
        self.assertEqual(len(tl), 152)
        self.assertEqual(tl[0], tf.start)
        prev = tf.start - timedelta(days = 1)
        for t in tl:
            self.assertGreater(t, prev)
            self.assertLessEqual(t, tf.end)
            prev = t


if __name__ == "__main__":
    unittest.main()
