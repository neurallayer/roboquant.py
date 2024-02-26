import roboquant as rq
from roboquant.journals import TensorboardJournal, PNLMetric, RunMetric, FeedMetric, PriceItemMetric, AlphaBeta
from tensorboard.summary import Writer

if __name__ == "__main__":
    """Compare 3 runs with different parameters using tensorboard"""

    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    params = [(3, 5), (13, 26), (12, 50)]

    for p1, p2 in params:
        s = rq.strategies.EMACrossover(p1, p2)
        log_dir = f"""runs/ema_{p1}_{p2}"""
        writer = Writer(log_dir)
        journal = TensorboardJournal(writer, PNLMetric(), RunMetric(), FeedMetric(), PriceItemMetric("JPM"), AlphaBeta(200))
        account = rq.run(feed, s, journal=journal)
        writer.close()
