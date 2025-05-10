# %%
import os
from timeit import default_timer as timer
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.parquet import ParquetFeed
from roboquant.feeds.sql import SQLFeed
from dotenv import load_dotenv


load_dotenv()
# %%
print("Retrieving historical data...")
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
alpaca_feed = AlpacaHistoricStockFeed(api_key, secret_key)
alpaca_feed.retrieve_quotes("AAPL", "GOOGL", start="2024-05-09T18:00:00Z", end="2024-05-09T21:00:00Z")
print("Original timeframe:", alpaca_feed.timeframe())
print("Original events:", alpaca_feed.count_events())
print("Original items:", alpaca_feed.count_items())

# %%
def print_results(feed: ParquetFeed | SQLFeed, file: str):
    try:
        os.remove(file)
    except OSError:
        pass
    print("############################")
    print("feed:", feed)
    start = timer()
    feed.record(alpaca_feed)
    recording_time = timer() - start
    print("timeframe:", feed.timeframe())
    print("events:", feed.count_events())
    print("items:", feed.count_items())
    print("file size:", os.path.getsize(file)//1024, "KB")
    print("recording time:", recording_time)

# %%
# Parquet recording
file = "/tmp/apple_quotes.parquet"
feed = ParquetFeed(file)
print_results(feed, file)

# %%
# SQLite recording
file = "/tmp/apple_quotes.db"
feed = SQLFeed(file, "quote")
print_results(feed, file)
