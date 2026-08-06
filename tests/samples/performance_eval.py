# %% [markdown]
# # Strategy Performance
# This example shows to compare a strategy against a common Buy & Hold strategy.
# %%
import roboquant as rq
from roboquant.journals.metricsjournal import MetricsJournal

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()

# %%
def run_and_plot(strategy):
    journal = MetricsJournal.pnl()
    account = rq.run(feed, strategy, journal = journal)
    print(account)
    ax = journal.plot("pnl/equity")
    ax.set_title(type(strategy).__name__ + " Equity")


# %%
run_and_plot(rq.strategies.EMACrossover());

# %%
run_and_plot(rq.strategies.BuyHoldStrategy(wait=26));
