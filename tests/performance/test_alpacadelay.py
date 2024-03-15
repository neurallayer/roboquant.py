import logging
import time
import unittest
from statistics import mean, stdev

from roboquant import Timeframe
from roboquant.feeds.alpacafeed import AlpacaLiveFeed


class TestAlpacaDelay(unittest.TestCase):

    def test_alpaca_delay(self):

        logging.basicConfig(level=logging.INFO)

        feed = AlpacaLiveFeed(market="iex")

        # subscribe to popular IEX stocks for Quotes
        feed.subscribe_quotes(
            "TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"
        )

        timeframe = Timeframe.next(minutes=1)
        channel = feed.play_background(timeframe, 1000)

        delays = []
        while event := channel.get(70):
            if event.items:
                delays.append(time.time() - event.time.timestamp())

        if delays:
            t = (
                f"mean={mean(delays):.3f} stdev={stdev(delays):.3f} "
                + f"max={max(delays):.3f} min={min(delays):.3f} n={len(delays)}"
            )
            print(t)
        else:
            print("didn't receive any items, is it perhaps outside trading hours?")


if __name__ == "__main__":
    unittest.main()
