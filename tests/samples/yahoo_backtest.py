# %% [markdown]
# This example shows how to use the Yahoo feed with a simple EMA Crossover strategy
# to run a backtest in roboquant.

# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("TSLA", "MSFT", "GOOG", start_date="2010-01-10")

# %%
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)

# %%
trades = sorted(account.trades, key=lambda t: t.pnl)
print(f"Biggest looser: {trades[0]}")
print(f"Biggest winner: {trades[-1]}")
