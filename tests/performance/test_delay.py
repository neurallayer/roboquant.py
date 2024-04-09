import time
import unittest
from statistics import mean, stdev

from roboquant import Timeframe
from roboquant.feeds import TiingoLiveFeed, Feed
from roboquant.alpaca import AlpacaLiveFeed


class TestDelay(unittest.TestCase):
    """
    Measure the delay of receiving live prices from IEX using Tiingo and Alpaca.

    This route includes the following paths if you run this script from home:

    - From IEX to the market data provider (Tiingo or Alpaca)
    - Provider holds it for 15ms (requirement from IEX)
    - From provider to the modem/access-point in your house
    - From the access-point to your computer (f.e lan or Wi-Fi)
    """

    def __get_symbols(self):
        # Popular stocks
        return ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]

    def __run_feed(self, feed: Feed):
        timeframe = Timeframe.next(minutes=1)
        channel = feed.play_background(timeframe, 1000)
        name = type(feed).__name__

        delays = []
        n = 0
        while event := channel.get(70):
            if event.items:
                n += len(event.items)
                delays.append(time.time() - event.time.timestamp())

        if delays:
            t = (
                f"feed={name} mean={mean(delays):.3f} stdev={stdev(delays):.3f} "
                + f"max={max(delays):.3f} min={min(delays):.3f} events={len(delays)} items={n}"
            )
            print(t)
        else:
            print(f"Didn't receive any items from {name}, is it perhaps outside trading hours?")

    def test_tiingo_delay(self):
        feed = TiingoLiveFeed(market="iex")
        feed.subscribe(*self.__get_symbols(), threshold_level=0)
        self.__run_feed(feed)
        feed.close()

    def test_alpaca_delay(self):
        feed = AlpacaLiveFeed(market="iex")
        feed.subscribe_quotes(*self.__get_symbols())
        self.__run_feed(feed)


if __name__ == "__main__":
    unittest.main()
