import time
import unittest
from statistics import mean, stdev

from roboquant import Timeframe
from roboquant.alpaca import AlpacaLiveFeed


class TestDelay(unittest.TestCase):
    """
    Measure the delay of receiving live prices from IEX using Alpaca.

    It requires that the system clock of your computer is set correctly.
    You can navigate to https://time.is/ to get a rough idea about its accuracy.

    This route includes the following paths if you run this script from home:

    - From IEX to the market data provider (Alpaca)
    - Provider holds it for 15ms (requirement from IEX)
    - From provider to the modem/access-point in your house
    - From the access-point to your computer (f.e lan or Wi-Fi)
    """

    __symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]

    def test_alpaca_delay(self):
        feed = AlpacaLiveFeed(market="iex")
        feed.subscribe_quotes(*TestDelay.__symbols)
        timeframe = Timeframe.next(minutes=1)
        channel = feed.play_background(timeframe, 1000)

        delays = []
        n = 0
        while event := channel.get(10):
            if event.items:
                n += len(event.items)
                delays.append(time.time() - event.time.timestamp())

        self.assertTrue(delays, "Didn't receive any quotes, is it perhaps outside trading hours?")

        print(
           f"delays mean={mean(delays):.3f} stdev={stdev(delays):.3f}",
           f"max={max(delays):.3f} min={min(delays):.3f} events={len(delays)} items={n}"
        )


if __name__ == "__main__":
    unittest.main()
