# %%
from datetime import timedelta

from roboquant.feeds import AggregatorFeed, get_sp500_symbols
from roboquant.alpaca import AlpacaLiveFeed

# %%
alpaca_feed = AlpacaLiveFeed()

stocks = get_sp500_symbols()[:30]
alpaca_feed.subscribe_quotes(*stocks)

# alpaca_feed.subscribe_bars(*stocks)
# feed.subscribe_trades("BTC/USD", "ETH/USD")
# feed.subscribe("SPXW240312C05190000")

feed = AggregatorFeed(alpaca_feed, timedelta(seconds=15), price_type="quote")

channel = feed.play_background()
while event := channel.get():
    print(event.items)
