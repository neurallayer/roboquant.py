import logging
import time
import unittest
from statistics import mean, stdev

from roboquant import Timeframe
from roboquant.feeds import TiingoLiveFeed


class TestTiingoDelay(unittest.TestCase):

    def test_tiingo_delay(self):
        """
        Measure the average delay receiving prices from IEX using Tiingo.
        This includes the following paths if you run it from home:

        - From IEX to Tiingo (New York)
        - Tiingo holds it for 15ms (requirement from IEX)
        - From Tiingo to the modem/access-point in your house
        - From the access-point to your computer (f.e lan or Wi-Fi)
        """

        logging.basicConfig(level=logging.INFO)

        feed = TiingoLiveFeed(market="iex")

        # subscribe to all IEX stocks for TOP of order book changes and Trades.
        feed.subscribe(threshold_level=5)

        timeframe = Timeframe.next(minutes=1)
        channel = feed.play_background(timeframe, 10_000)

        delays = []
        while event := channel.get():
            if event.items:
                delays.append(time.time() - event.time.timestamp())

        if delays:
            t = (
                f"mean={mean(delays):.3f} stdev={stdev(delays):.3f} "
                + f"max={max(delays):.3f} min={min(delays):.3f} n={len(delays)}"
            )
            print(t)
        else:
            print("didn't receive any actions, is it perhaps outside trading hours?")

        feed.close()


if __name__ == "__main__":
    unittest.main()
