# %%
from datetime import timedelta
from roboquant.feeds import AggregatorFeed, AlpacaLiveFeed, get_sp500_symbols

# %%
alpaca_feed = AlpacaLiveFeed()
# feed.subscribe_trades("BTC/USD", "ETH/USD")
stocks = get_sp500_symbols()[:30]
alpaca_feed.subscribe_quotes(*stocks)
# alpaca_feed.subscribe_bars(*stocks)

# feed.subscribe("SPXW240312C05190000")
feed = AggregatorFeed(alpaca_feed,  timedelta(seconds=15), item_type="quote")

channel = feed.play_background()
while event := channel.get():
    print(event.items)
