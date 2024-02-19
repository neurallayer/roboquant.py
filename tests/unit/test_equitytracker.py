import unittest
from roboquant import Roboquant, EquityTracker, EMACrossover
from tests.common import get_feed


class TestCAPMTracker(unittest.TestCase):

    def test_capmtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()
        tracker = EquityTracker()
        rq.run(feed, tracker=tracker)

        timeline, equity = tracker.timeline, tracker.equity

        self.assertEqual(len(timeline), len(equity))
        self.assertEqual(feed.timeframe().start, timeline[0])
        self.assertEqual(feed.timeframe().end, timeline[-1])
        self.assertEqual(1_000_000.0, equity[0])


if __name__ == "__main__":
    unittest.main()
