import unittest
from roboquant import Roboquant, EMACrossover, StandardTracker
from tests.common import get_feed, get_output


class StandardTrackerTest(unittest.TestCase):

    def test_standardtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()
        tracker = StandardTracker()
        rq.run(feed, tracker=tracker)

        self.maxDiff = None
        self.assertEqual(tracker.__repr__(), get_output("standardtracker_repr.txt"))


if __name__ == "__main__":
    unittest.main()
