# %%
import roboquant as rq
from roboquant.feeds.parquet import ParquetFeed

# %%
yahoo_feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "MSFT", "TSLA", start_date="2000-01-01")

# %%
feed = ParquetFeed("/tmp/stocks.parquet")
if not feed.exists():
    feed.record(yahoo_feed)
