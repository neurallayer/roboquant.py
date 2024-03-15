import json
import pathlib
from datetime import datetime

from roboquant.event import Bar
from roboquant.feeds.feed import Feed
from roboquant.timeframe import Timeframe


def get_symbol_prices(
        feed: Feed, symbol: str, price_type="DEFAULT", timeframe: Timeframe | None = None
) -> tuple[list[datetime], list[float]]:
    """Get prices for a single symbol from a feed"""

    x = []
    y = []
    channel = feed.play_background(timeframe)
    while event := channel.get():
        price = event.get_price(symbol, price_type)
        if price:
            x.append(event.time)
            y.append(price)
    return x, y


def get_symbol_ohlcv(feed: Feed, symbol: str, timeframe: Timeframe | None = None) -> dict[str, list]:
    """Get the OHLCV values for a single symbol from a feed"""

    result = {column: [] for column in ["Date", "Open", "High", "Low", "Close", "Volume"]}
    channel = feed.play_background(timeframe)
    while event := channel.get():
        item = event.price_items.get(symbol)
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
