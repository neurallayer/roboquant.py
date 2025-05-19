# %%
import os
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed

# %%
feed = ParquetFeed("/tmp/us_nasdaq.parquet")
if not feed.exists():
    print("Recording feed...")
    path = os.path.expanduser("~/data/daily/us/nasdaq stocks/")
    csv_feed = rq.feeds.CSVFeed.stooq_us_daily(path)
    feed.record(csv_feed, row_group_size=100_000)

# %%
print("Starting backtest...")
account = rq.run(feed, rq.strategies.EMACrossover())
print(account)
