import pathlib
import unittest

from roboquant.feeds import CSVFeed
from roboquant.journals import RunMetric
from roboquant.strategies.emacrossover import EMACrossover
from roboquant.run import run
from roboquant.journals.plotjournal import PlotJournal


class TestJournal(unittest.TestCase):

    @staticmethod
    def _get_root_dir(*paths):
        root = pathlib.Path(__file__).parent.resolve().joinpath("..", "data", *paths)
        return str(root)

    def test_plot_journal(self):
        root = self._get_root_dir("yahoo")
        feed = CSVFeed.yahoo(root)
        apple = feed.get_asset("AAPL")
        strategy = EMACrossover()
        journal = PlotJournal(apple, RunMetric())

        run(feed, strategy, journal=journal)
        journal.plot()


if __name__ == "__main__":
    unittest.main()
