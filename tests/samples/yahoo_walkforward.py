# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed.us_stocks_10(start_date="2000-01-01", end_date="2020-01-01")

# %%
# split the feed timeframe into 4 parts
timeframes = feed.timeframe().split(4)

# %%
# run a walkforward back-test on each timeframe
for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe.strftime("%Y-%m-%d")}  equity={account.equity():.0f}")
