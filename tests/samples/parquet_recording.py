# %%
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed

# %%
yahoo_feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "MSFT", "TSLA", start_date="2000-01-01")

# %%
feed = ParquetFeed("/tmp/stocks.parquet")
if not feed.exists():
    feed.record(yahoo_feed)

# %%
# split the feed timeframe in 4 equal parts
timeframes = feed.timeframe().split(4)

# %%
# run a walkforward back-test on each timeframe
for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe}  equity={account.equity()}")