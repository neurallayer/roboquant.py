
from datetime import timedelta
from roboquant.feeds.aggregate import AggregatorFeed
from roboquant.feeds.alpacafeed import AlpacaLiveFeed
from roboquant.feeds.feedutil import get_sp500_symbols


def run():
    alpaca_feed = AlpacaLiveFeed()
    # feed.subscribe_trades("BTC/USD", "ETH/USD")
    stocks = get_sp500_symbols()[:30]
    alpaca_feed.subscribe_quotes(*stocks)

    # feed.subscribe("SPXW240312C05190000")
    feed = AggregatorFeed(alpaca_feed,  timedelta(seconds=15), item_type="quote")

    channel = feed.play_background()
    while event := channel.get(30.0):
        print(event)


if __name__ == "__main__":
    run()
