import unittest

from roboquant.journals import RunMetric, MetricsJournal, FeedMetric, MarketMetric, PNLMetric
from roboquant.strategies.emacrossover import EMACrossover
from roboquant.run import run
from roboquant.journals.scorecard import ScoreCard
from tests.common import get_feed


class TestJournal(unittest.TestCase):

    def test_scorecard(self):
        feed = get_feed()
        strategy = EMACrossover()
        journal = ScoreCard(RunMetric())
        run(feed, strategy, journal=journal)
        journal.plot()

    def test_metrics(self):
        feed = get_feed()
        strategy = EMACrossover()
        journal = MetricsJournal(RunMetric(), FeedMetric(), MarketMetric(), PNLMetric())
        run(feed, strategy, journal=journal)
        self.assertTrue(journal.get_metric_names())
        self.assertEqual(1218, len(journal.get_metric("pnl/equity")[0]))

        df = journal.to_dataframe("pnl/equity")
        self.assertEqual(1218, len(df))


if __name__ == "__main__":
    unittest.main()
