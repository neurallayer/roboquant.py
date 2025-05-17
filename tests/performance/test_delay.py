import os
import time
import unittest
from statistics import mean, stdev
from dotenv import load_dotenv

from roboquant import Timeframe
from roboquant.alpaca import AlpacaLiveFeed

load_dotenv()


class TestDelay(unittest.TestCase):
    """
    Measure the delay of receiving live market data using Alpaca. It will take two times one minute to run and will
    collect the quotes for a number of populair stocks and crypto currencies. The delay is the time between the moment
    the quote was generated on the exchange and the moment it was received by your system.

    It requires that the system clock of your computer is set correctly.
    You can navigate to https://time.is/ to get a rough idea about its accuracy.

    This route includes the following paths if you run this script from home:

    - From exchange to the market data provider (Alpaca)
    - Provider holds it for 15ms in case of IEX stock data
    - From provider to the modem/access-point in your house
    - From the access-point to your computer (f.e lan or Wi-Fi)
    """

    def _measure(self, title, feed):
        print()
        print(title)
        print("=" * len(title))

        timeframe = Timeframe.next(minutes=1)

        delays = []
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

    def test_alpaca_delay_stocks(self):
        symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]

        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET"]
        feed = AlpacaLiveFeed(api_key, secret_key, market="iex")
        feed.subscribe_quotes(*symbols)
        self._measure("IEX Exchange Delay", feed)

    def test_alpaca_delay_crypto(self):
        cryptos = ["BTC", "ETH", "XRB", "BNB", "SOL", "DOGE"]
        symbols = [f"{c}/USD" for c in cryptos] + [f"{c}/USDT" for c in cryptos] + [f"{c}/USDC" for c in cryptos]

        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET"]
        feed = AlpacaLiveFeed(api_key, secret_key, market="crypto")
        feed.subscribe_quotes(*symbols)
        self._measure("Crypto Exchange Delay", feed)


if __name__ == "__main__":
    unittest.main()
