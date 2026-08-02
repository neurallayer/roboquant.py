# %% [markdown]
# This example shows to compare a strategy against a common Buy & Hold strategy.
# %%
import roboquant as rq
from roboquant.journals.metricsjournal import MetricsJournal
from roboquant.traders.simpletrader import SimpleTrader

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()

# %%
def run_and_plot(strategy):
    trader = SimpleTrader()
    journal = MetricsJournal.pnl()
    account = rq.run(feed, strategy, journal = journal, trader=trader)
    print(account)
    journal.plot("pnl/equity")


# %%
run_and_plot(rq.strategies.EMACrossover());

# %%
run_and_plot(rq.strategies.BuyHoldStrategy(wait=26));
