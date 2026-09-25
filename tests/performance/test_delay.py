from datetime import timedelta
import os
import time
from statistics import mean, stdev
from dotenv import load_dotenv

from roboquant import Timeframe
from roboquant.feeds import Feed
from roboquant.feeds.alpaca import AlpacaLiveFeed

load_dotenv()


def _measure(title: str, feed: Feed):
    print()
    print(title)
    print("=" * len(title))

    timeframe = Timeframe.next(timedelta(minutes=1))

    delays: list[float] = []
    n = 0
    for event in feed.play(timeframe):
        if event.items:
            n += len(event.items)
            delays.append(time.time() - event.time.timestamp())

    if not delays:
        print("Didn't receive any quotes, is it perhaps outside trading hours?")
    else:
        print(
            f"delays mean={mean(delays):.3f} stdev={stdev(delays):.3f}",
            f"max={max(delays):.3f} min={min(delays):.3f} events={len(delays)} items={n}",
        )

def test_alpaca_delay_stocks():
    symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]

    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET"]
    feed = AlpacaLiveFeed(api_key, secret_key, market="iex")
    feed.subscribe_quotes(*symbols)
    _measure("IEX Exchange Delay", feed)

def test_alpaca_delay_crypto():
    cryptos = ["BTC", "ETH", "XRB", "BNB", "SOL", "DOGE"]
    symbols = [f"{c}/USD" for c in cryptos] + [f"{c}/USDT" for c in cryptos] + [f"{c}/USDC" for c in cryptos]

    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET"]
    feed = AlpacaLiveFeed(api_key, secret_key, market="crypto")
    feed.subscribe_quotes(*symbols)
    _measure("Crypto Exchange Delay", feed)


if __name__ == "__main__":
    test_alpaca_delay_stocks()
    test_alpaca_delay_crypto()

