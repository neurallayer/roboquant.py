# %%
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed


# %%
feed = ParquetFeed("/tmp/us_large.parquet")
if not feed.exists():
    print("recording feed...")
    csv_feed = rq.feeds.CSVFeed.stooq_us_daily("/tmp/us")
    feed.record(csv_feed, row_group_size=100_000)

# %%
print("back test...")
account = rq.run(feed, rq.strategies.EMACrossover())
print(account)
