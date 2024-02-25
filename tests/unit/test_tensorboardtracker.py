from pathlib import Path
import tempfile
import unittest
from roboquant import Roboquant
from roboquant.strategies import EMACrossover
from roboquant.trackers import RunMetric, EquityMetric, TensorboardTracker
from tests.common import get_feed
from torch.utils.tensorboard.writer import SummaryWriter


class TestTensorboardTracker(unittest.TestCase):

    def test_tensorboardtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()

        tmpdir = tempfile.gettempdir()

        log_dir = Path(tmpdir).joinpath("runs")
        writer = SummaryWriter(log_dir)
        tracker = TensorboardTracker(writer, RunMetric(), EquityMetric())
        rq.run(feed, tracker=tracker)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
