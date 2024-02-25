from pathlib import Path
import tempfile
import unittest
from roboquant import Roboquant
from roboquant.strategies import EMACrossover
from roboquant.trackers import RunMetric, EquityMetric, TensorboardTracker
from tests.common import get_feed
from tensorboard.summary import Writer


class TestTensorboardTracker(unittest.TestCase):

    def test_tensorboardtracker(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()

        tmpdir = tempfile.gettempdir()

        output = Path(tmpdir).joinpath("runs")
        writer = Writer(str(output))
        tracker = TensorboardTracker(writer, RunMetric(), EquityMetric())
        rq.run(feed, tracker=tracker)
        writer.close()


if __name__ == "__main__":
    unittest.main()
