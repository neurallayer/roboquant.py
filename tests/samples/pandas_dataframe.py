# %%
import pandas as pd
import roboquant as rq

pd.set_option('display.width', 1000)
# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
df = feed.to_timeseries(rq.Stock("IBM"))
print("IBM Stock prices", df, sep="\n")

# %%
feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "TSLA", "INTC", start_date="2020-01-01")
data = feed.to_timeseries(*feed.assets())
print("Asset correlations:\n", data.corr())


# %%
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)

#%%
print(account.portfolio.to_dataframe(), "\n\n")
print(account.orders_to_dataframe(), "\n\n")
print(account.trades_to_dataframe(), "\n\n")

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)
equity = journal.get_metrics("pnl/equity")
print("Equity", equity, sep="\n")
