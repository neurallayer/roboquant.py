# %%
import os
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.avro import AvroFeed
from dotenv import load_dotenv
load_dotenv()
# %%
print("The retrieval of historical data will take some time and will require plenty of free diskspace")
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
alpaca_feed = AlpacaHistoricStockFeed(api_key, secret_key)
alpaca_feed.retrieve_quotes("AAPL", start="2024-05-09T18:00:00Z", end="2024-05-09T19:00:00Z")

# %%
feed = AvroFeed("/tmp/apple_quotes.avro")
print(alpaca_feed)

# %%
# store it for later use
feed.record(alpaca_feed)

# %%
# Info on recorded feed
print(feed.timeframe())
print(feed.count_events())
