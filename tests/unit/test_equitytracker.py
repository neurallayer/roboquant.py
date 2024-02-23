import unittest
from roboquant import Roboquant, EquityTracker, EMACrossover
from tests.common import get_feed


class TestEquityTracker(unittest.TestCase):

    def test_equitytracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()
        tracker = EquityTracker()
        rq.run(feed, tracker=tracker)

        timeline, equity = tracker.timeline, tracker.equities

        self.assertEqual(len(timeline), len(equity))
        self.assertEqual(feed.timeframe().start, timeline[0])
        self.assertEqual(feed.timeframe().end, timeline[-1])
        self.assertEqual(1_000_000.0, equity[0])

        mdd = tracker.max_drawdown()
        self.assertTrue(-10 < mdd < 0)

        gain = tracker.max_gain()
        self.assertTrue(0 < gain < 10)

        pnl = tracker.pnl(True)
        self.assertTrue(-1 < pnl < 1)

if __name__ == "__main__":
    unittest.main()
