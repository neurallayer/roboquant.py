# %%
from roboquant.alpaca import AlpacaLiveFeed
from roboquant import Timeframe

# %%
feed = AlpacaLiveFeed()

symbols = ["F", "TSLA", "MSFT"]
feed.subscribe_quotes(*symbols)

# alpaca_feed.subscribe_bars(*stocks)
# feed.subscribe_trades("BTC/USD", "ETH/USD")
# feed.subscribe("SPXW240312C05190000")

timeframe = Timeframe.next(minutes=1)
channel = feed.play_background(timeframe)
while event := channel.get():
    print(event.time, event.items)
