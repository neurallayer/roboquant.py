# %%
from tensorboard.summary import Writer
import roboquant as rq
from roboquant.journals import TensorboardJournal, PNLMetric, RunMetric, AlphaBeta

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "MSFT", "V", "GE", "CSCO", "WMT", "XOM", "INTC", start_date="2010-01-01")

# Compare runs with different parameters for the EMACrossover strategy
hyper_params = [(5, 10), (12, 25), (25, 50)]

for p1, p2 in hyper_params:

    # Each run will be logged to a different directory
    log_dir = f"runs/ema_{p1}_{p2}"
    writer = Writer(log_dir)

    strategy = rq.strategies.EMACrossover(p1, p2)
    journal = TensorboardJournal(writer, PNLMetric(), RunMetric(), AlphaBeta(200))
    account = rq.run(feed, strategy, journal=journal)
    print(p1, p2, account.equity())
    writer.close()
