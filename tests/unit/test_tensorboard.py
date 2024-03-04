import tempfile
import unittest
from pathlib import Path

from tensorboard.summary import Writer

import roboquant as rq
from roboquant.journals import RunMetric, PNLMetric, TensorboardJournal
from tests.common import get_feed


class TestTensorboard(unittest.TestCase):

    def test_tensorboard_journal(self):
        feed = get_feed()

        tmpdir = tempfile.gettempdir()

        output = Path(tmpdir).joinpath("runs")
        writer = Writer(str(output))
        journal = TensorboardJournal(writer, RunMetric(), PNLMetric())
        rq.run(feed, rq.strategies.EMACrossover(), journal=journal)
        writer.close()


if __name__ == "__main__":
    unittest.main()
