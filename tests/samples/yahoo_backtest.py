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
df = account.trades_to_dataframe().round(2)
print(f"Big looser: {df[df.pnl < -200_000]}")
print(f"Big winner: {df[df.pnl > 200_000]}")
