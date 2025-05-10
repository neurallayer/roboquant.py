# %%
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed

# %%
csv_feed = rq.feeds.CSVFeed.stooq_us_daily("/tmp/us")

# 10 popular symbols
symbols_str = "MSFT,NVDA,AAPL,AMZN,META,GOOGL,AVGO,JPM,XOM,TSLA"
symbols = set(symbols_str.split(","))

def symbol_filter(item: rq.PriceItem):
    return item.asset.symbol in symbols

# 10 years
tf = rq.Timeframe.fromisoformat("2015-01-01T00:00:00", "2025-01-01T00:00:00")
# %%
feed = ParquetFeed("/tmp/us10.parquet")
feed.record(csv_feed, tf, priceitem_filter=symbol_filter)
