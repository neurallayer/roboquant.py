# %%
import os
from roboquant.third_party.alpaca import AlpacaLiveFeed
from roboquant import Timeframe
from dotenv import load_dotenv
load_dotenv()

# %%
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]

feed = AlpacaLiveFeed(api_key, secret_key)
symbols = ["F", "TSLA", "MSFT"]
feed.subscribe_quotes(*symbols)

# feed = AlpacaLiveFeed(api_key, secret_key, market="crypto")
# feed.subscribe_trades("BTC/USD", "ETH/USD")

timeframe = Timeframe.next("1 minute")
for event in feed.play(timeframe):
    if event.is_empty():
        print("Are you sure the market is open?")
    else:
        print(event.time, event.items)
