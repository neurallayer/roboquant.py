from pathlib import Path
import tempfile
import unittest
from roboquant import Roboquant, EMACrossover
from roboquant.trackers.tensorboardtracker import TensorboardTracker
from tests.common import get_feed
from torch.utils.tensorboard.writer import SummaryWriter


class TestTensorboardTracker(unittest.TestCase):

    def test_tensorboardtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()

        tmpdir = tempfile.gettempdir()

        log_dir = Path(tmpdir).joinpath("runs")
        writer = SummaryWriter(log_dir)
        tracker = TensorboardTracker(writer)
        rq.run(feed, tracker=tracker)
        self.assertGreater(tracker.items, 0)
        self.assertGreater(tracker.ratings, 0)
        self.assertGreater(tracker.orders, 0)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
