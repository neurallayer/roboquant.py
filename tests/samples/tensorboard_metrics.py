# %%
from tensorboard.summary import Writer

import roboquant as rq
from roboquant.journals import TensorboardJournal, PNLMetric, RunMetric, PriceItemMetric, AlphaBeta

# %%
# Compare runs with different parameters using tensorboard
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "MSFT", "V", "GE", "CSCO", "WMT", "XOM", "INTC", start_date="2010-01-01")

hyper_params = [(3, 5), (13, 26), (12, 50)]

for p1, p2 in hyper_params:
    s = rq.strategies.EMACrossover(p1, p2)
    log_dir = f"runs/ema_{p1}_{p2}"
    writer = Writer(log_dir)
    journal = TensorboardJournal(writer, PNLMetric(), RunMetric(), PriceItemMetric("JPM"), AlphaBeta(200))
    account = rq.run(feed, s, journal=journal)
    print(p1, p2, account.equity())
    writer.close()
