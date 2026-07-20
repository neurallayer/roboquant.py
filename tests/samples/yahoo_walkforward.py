# %% [hidden]
import roboquant as rq

# %% [markdown]
# First we need to get the data for our backtest. It should have sufficient
# historic data for each step in our walkforward.
# <br/><br/>
# In this example we get over 25 years of historic market data from Yahoo Finance

# %%
feed = rq.feeds.YahooFeed.us_stocks_10(start_date="2000-01-01")

# %% [markdown]
# Now we need to decide on the timeframe for each step in the walkforward backtest.
# An easy way to achieve this, is to get the total timeframe of the feed and split it into
# `n` equal lenght timeframes.
# %%
timeframes = feed.timeframe().split(5)

# %% [markdown]
# We can now iterate over these timeframes and limit each run to the timeframe.
# %%
for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe.strftime("%Y-%m-%d")}  equity={account.equity():.0f}")

# %%
