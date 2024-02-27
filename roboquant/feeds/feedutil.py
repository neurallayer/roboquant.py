import threading
from datetime import datetime

from roboquant.event import Candle
from roboquant.timeframe import Timeframe
from .eventchannel import EventChannel, ChannelClosed
from .feed import Feed


def play_background(feed: Feed, channel: EventChannel):
    """Play a feed in the background on its own thread.
    The provided channel will be closed after the playing has finished.
    """

    def __background():
        try:
            feed.play(channel)
        except ChannelClosed:
            """this exception we can expect"""
        finally:
            channel.close()

    thread = threading.Thread(None, __background, daemon=False)
    thread.start()


def get_symbol_prices(
        feed: Feed, symbol: str, price_type="DEFAULT", timeframe: Timeframe | None = None
) -> tuple[list[datetime], list[float]]:
    """Get prices for a single symbol from a feed"""

    x = []
    y = []
    channel = EventChannel(timeframe)
    play_background(feed, channel)
    while event := channel.get():
        price = event.get_price(symbol, price_type)
        if price:
            x.append(event.time)
            y.append(price)
    return x, y


def get_symbol_ohlcv(feed: Feed, symbol: str, timeframe: Timeframe | None = None) -> list[tuple]:
    """Get the candles for a single symbol from a feed"""

    result = []
    channel = EventChannel(timeframe)
    play_background(feed, channel)
    while event := channel.get():
        item = event.price_items.get(symbol)
        if item and isinstance(item, Candle):
            result.append((event.time, *item.ohlcv))
    return result


def get_symbol_dataframe(feed: Feed, symbol: str, timeframe: Timeframe | None = None):
    """Get prices for a single symbol from a feed as a pandas dataframe"""

    # noinspection PyPackageRequirements
    import pandas as pd

    ohlcv = get_symbol_ohlcv(feed, symbol, timeframe)
    return pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")


def print_feed_items(feed: Feed, timeframe: Timeframe | None = None, timeout: float | None = None):
    """Print the items in a feed to the console.
    This is mostly useful for debugging purposes to see what events a feed generates.
    """

    channel = EventChannel(timeframe)
    play_background(feed, channel)
    while event := channel.get(timeout):
        print(event.time)
        for item in event.items:
            print("======> ", item)
