# %%
import os
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed


# %%
# 10 popular symbols
symbols_str = "MSFT,NVDA,AAPL,AMZN,META,GOOGL,AVGO,JPM,XOM,TSLA"
symbols = set(symbols_str.split(","))

def asset_filter(asset: rq.Asset):
    return asset.symbol in symbols

path = os.path.expanduser("~/data/daily/us/")
csv_feed = rq.feeds.CSVFeed.stooq_us_daily(path, asset_filter=asset_filter)
csv_feed_symbols = {a.symbol for a in csv_feed.assets()}
assert csv_feed_symbols == symbols

# %%
# 10 years
tf = rq.Timeframe.fromisoformat("2015-01-01T00:00:00", "2025-01-01T00:00:00")
feed = ParquetFeed("/tmp/us10.parquet")
feed.record(csv_feed, tf)
