import pathlib
import unittest

from roboquant.feeds import CSVFeed
from roboquant.journals import RunMetric, MetricsJournal, FeedMetric, MarketMetric, PNLMetric
from roboquant.strategies.emacrossover import EMACrossover
from roboquant.run import run
from roboquant.journals.chartingjournal import ChartingJournal


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
        journal = ChartingJournal(apple, RunMetric())

        run(feed, strategy, journal=journal)
        journal.plot()

    def test_metrics(self):
        root = self._get_root_dir("yahoo")
        feed = CSVFeed.yahoo(root)
        # apple = feed.get_asset("AAPL")
        strategy = EMACrossover()
        journal = MetricsJournal(RunMetric(), FeedMetric(), MarketMetric(), PNLMetric())
        run(feed, strategy, journal=journal)


if __name__ == "__main__":
    unittest.main()
