from pathlib import Path
import tempfile
import unittest
from roboquant import Roboquant
from roboquant.strategies import EMACrossover
from roboquant.journals import RunMetric, EquityMetric, TensorboardJournal
from tests.common import get_feed
from tensorboard.summary import Writer


class TestTensorboard(unittest.TestCase):

    def test_tensorboard_journal(self):
        rq = Roboquant(EMACrossover())
        feed = get_feed()

        tmpdir = tempfile.gettempdir()

        output = Path(tmpdir).joinpath("runs")
        writer = Writer(str(output))
        journal = TensorboardJournal(writer, RunMetric(), EquityMetric())
        rq.run(feed, journal=journal)
        writer.close()


if __name__ == "__main__":
    unittest.main()
