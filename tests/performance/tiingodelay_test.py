"""
Measure the average delay receiving prices from IEX. This includes the following paths:

- From IEX to Tiingo
- Tiingo holds it for 15ms (requirement from IEX)
- From Tiingo to the modem/access-point in your house
- From the access-point to your computer (f.e lan or Wi-Fi)
"""

import logging
import time

from roboquant import EventChannel, Timeframe, TiingoLiveFeed, feedutil
from statistics import mean, stdev

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    feed = TiingoLiveFeed(market="iex")

    # subscribe to all IEX stocks for TOP of order book changes and Trades.
    feed.subscribe(threshold_level=5)

    timeframe = Timeframe.next(minutes=1)
    channel = EventChannel(timeframe, maxsize=10_000)
    feedutil.play_background(feed, channel)

    delays = []
    while event := channel.get():
        if event.items:
            delays.append(time.time() - event.time.timestamp())

    if delays:
        print(f"mean={mean(delays):.3f} stdev={stdev(delays):.3f} max={max(delays):.3f} min={min(delays):.3f} n={len(delays)}")
    else:
        print("didn't receive any actions, is it perhaps outside trading hours?")

    feed.close()
