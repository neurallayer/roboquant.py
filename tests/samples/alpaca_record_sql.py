# %%
import os
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.sql import SQLFeed
from dotenv import load_dotenv
load_dotenv()

# %%
feed = SQLFeed("/tmp/apple_quotes.db", "quote")

if not feed.exists():
    print("The retrieval of historical data will take some time....")
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET"]
    alpaca_feed = AlpacaHistoricStockFeed(api_key, secret_key)
    alpaca_feed.retrieve_quotes("AAPL", start="2024-05-09T00:00:00Z", end="2024-05-10T00:00:00Z")
    print(alpaca_feed)

    # store it for later use
    feed.record(alpaca_feed)

# %%
# Info on recorded feed
print(feed)
print(feed.count_events())
