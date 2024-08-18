# %%
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.parquet import ParquetFeed

# %%
print("The retrieval of historical data will take some time....")
alpaca_feed = AlpacaHistoricStockFeed()
alpaca_feed.retrieve_quotes("AAPL", start="2024-05-09T18:00:00Z", end="2024-05-09T19:00:00Z")
print(alpaca_feed)

# %%
# store it for later use
feed = ParquetFeed("/tmp/apple_quotes.parquet")
feed.record(alpaca_feed)

# %%
# Info on recorded feed
print(feed.timeframe())
print(feed.count_events())
