from roboquant import (
    Roboquant,
    EMACrossover,
    YahooFeed,
    TensorboardTracker,
    EquityMetric,
    RunMetric,
    FeedMetric,
    PriceItemMetric,
)
from torch.utils.tensorboard.writer import SummaryWriter

if __name__ == "__main__":
    """Compare 3 runs with different parameters using tensorboard"""

    feed = YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    params = [(3, 5), (13, 26), (12, 50)]

    for p1, p2 in params:
        rq = Roboquant(EMACrossover(p1, p2))
        log_dir = f"""runs/ema_{p1}_{p2}"""
        writer = SummaryWriter(log_dir)
        tracker = TensorboardTracker(writer, EquityMetric(), RunMetric(), FeedMetric(), PriceItemMetric("JPM"))
        account = rq.run(feed, tracker)
