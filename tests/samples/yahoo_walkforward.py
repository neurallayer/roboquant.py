# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01", end_date="2020-01-01")

# %%
# split the feed timeframe into 4 parts
timeframes = feed.timeframe().split(4)

# %%
# run a walkforward back-test on each timeframe
for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe}  equity={account.equity()}")
