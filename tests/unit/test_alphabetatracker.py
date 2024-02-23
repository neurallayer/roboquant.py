import unittest
from roboquant import Roboquant
from roboquant.strategies import EMACrossover
from roboquant.trackers import AlphaBetaTracker
from tests.common import get_feed


class TestCAPMTracker(unittest.TestCase):

    def test_capmtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()
        tracker = AlphaBetaTracker()
        rq.run(feed, tracker=tracker)
        alpha, beta = tracker.alpha_beta()
        self.assertGreater(alpha, -1)
        self.assertTrue(0 < beta < 1)


if __name__ == "__main__":
    unittest.main()
