# %% [markdown]
# This example shows how to use the Yahoo feed with a simple EMA Crossover strategy.

# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "TSLA", "F", "INTC", start_date="2015-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)

# %%
trades = sorted(account.trades, key=lambda t: t.pnl)
print(f"Biggest looser: {trades[0]}")
print(f"Biggest winner: {trades[-1]}")
