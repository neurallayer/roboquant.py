import unittest

from roboquant.journals import MetricsJournal
from roboquant.journals.metrics import AssetMetric, MarketMetric, PNLMetric, RunMetric
from roboquant.strategies.ema_crossover import EMACrossover
from roboquant.run import run
from roboquant.journals.scorecard import Scorecard
from tests.common import get_feed


class TestJournal(unittest.TestCase):

    def test_scorecard(self):
        feed = get_feed()
        strategy = EMACrossover()
        journal = Scorecard(RunMetric())
        run(feed, strategy, journal=journal)

    def test_metrics_journal(self):
        feed = get_feed()
        strategy = EMACrossover()
        journal = MetricsJournal(RunMetric(), AssetMetric(), MarketMetric(), PNLMetric())
        run(feed, strategy, journal=journal)
        self.assertTrue(journal.get_metric_names())
        equity = journal.get_metric("pnl/equity")
        self.assertEqual(1218, len(equity))


if __name__ == "__main__":
    unittest.main()
