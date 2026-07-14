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

    def test_metrics_journal(self):
        feed = get_feed()
        strategy = EMACrossover()
        journal = MetricsJournal(RunMetric(), FeedMetric(), MarketMetric(), PNLMetric())
        run(feed, strategy, journal=journal)
        self.assertTrue(journal.get_metric_names())
        equity = journal.get_metric("pnl/equity")
        self.assertEqual(1218, len(equity))
        df = equity.to_dataframe()
        self.assertEqual(1218, len(df))
        self.assertEqual("pnl/equity", df.Name)


if __name__ == "__main__":
    unittest.main()
