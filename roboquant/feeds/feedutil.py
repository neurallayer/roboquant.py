import json
import pathlib

from roboquant.asset import Asset
from roboquant.event import Bar
from roboquant.feeds.feed import Feed
from roboquant.timeframe import Timeframe


def get_ohlcv(feed: Feed, asset: Asset, timeframe: Timeframe | None = None) -> dict[str, list]:
    """Get the OHLCV values for a asset from a feed.
    The returned value is a dict with the keys being "Date", "Open", "High", "Low", "Close", "Volume"
    and the values a list.
    """

    result = {column: [] for column in ["Date", "Open", "High", "Low", "Close", "Volume"]}
    channel = feed.play_background(timeframe)
    while event := channel.get():
        item = event.price_items.get(asset)
        if item and isinstance(item, Bar):
            result["Date"].append(event.time)
            result["Open"].append(item.ohlcv[0])
            result["High"].append(item.ohlcv[1])
            result["Low"].append(item.ohlcv[2])
            result["Close"].append(item.ohlcv[3])
            result["Volume"].append(item.ohlcv[4])
    return result


def get_sp500_symbols():
    full_path = pathlib.Path(__file__).parent.resolve().joinpath("sp500.json")
    with open(full_path, encoding="utf8") as f:
        content = f.read()
        j = json.loads(content)
        symbols = [elem["Symbol"] for elem in j]
        return symbols


def print_feed_items(feed: Feed, timeframe: Timeframe | None = None, timeout: float | None = None):
    """Print the items in a feed to the console.
    This is mostly useful for debugging purposes to see what events a feed generates.
    """

    channel = feed.play_background(timeframe)
    while event := channel.get(timeout):
        print(event.time)
        for item in event.items:
            print("======> ", item)


def count_events(feed: Feed, timeframe: Timeframe | None = None, timeout: float | None = None, include_empty=False):
    """Count the number of events in a feed"""

    channel = feed.play_background(timeframe)
    events = 0
    while evt := channel.get(timeout):
        if evt.items or include_empty:
            events += 1
    return events


def count_items(feed: Feed, timeframe: Timeframe | None = None, timeout: float | None = None):
    """Count the number of events in a feed"""

    channel = feed.play_background(timeframe)
    items = 0
    while evt := channel.get(timeout):
        items += len(evt.items)
    return items
