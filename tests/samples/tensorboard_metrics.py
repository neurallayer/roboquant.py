from roboquant import Roboquant
from roboquant.feeds import YahooFeed
from roboquant.strategies import EMACrossover
from roboquant.journals import TensorboardJournal, EquityMetric, RunMetric, FeedMetric, PriceItemMetric, AlphaBeta
from tensorboard.summary import Writer

if __name__ == "__main__":
    """Compare 3 runs with different parameters using tensorboard"""

    feed = YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    params = [(3, 5), (13, 26), (12, 50)]

    for p1, p2 in params:
        rq = Roboquant(EMACrossover(p1, p2))
        log_dir = f"""runs/ema_{p1}_{p2}"""
        writer = Writer(log_dir)
        journal = TensorboardJournal(writer, EquityMetric(), RunMetric(), FeedMetric(), PriceItemMetric("JPM"), AlphaBeta(200))
        account = rq.run(feed, journal)
        writer.close()
