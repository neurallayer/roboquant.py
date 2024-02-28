from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any


@dataclass(slots=True)
class PriceItem:
    """Different type of price-times subclass this class"""

    symbol: str
    """the symbol for this price-item"""

    def price(self, price_type: str = "DEFAULT") -> float:
        """Returns the price for the provided price_type.
        A price_type, for example, is `OPEN` or `CLOSE`.

        All price-items are expected to return a DEFAULT price if the type is unknown.
        """
        ...

    def volume(self, volume_type: str = "DEFAULT") -> float:
        """Return the volume of the price-item"""
        ...


@dataclass(slots=True)
class Quote(PriceItem):
    data: array

    def price(self, price_type: str = "DEFAULT") -> float:
        match price_type:
            case "ASK":
                return self.data[0]
            case "BID":
                return self.data[2]
            case _:
                # Default is the mid-point price
                return (self.data[0] + self.data[2]) / 2.0

    @property
    def ask_volume(self) -> float:
        return self.data[1]

    @property
    def bid_volume(self) -> float:
        return self.data[3]

    def volume(self, volume_type: str = "DEFAULT") -> float:
        match volume_type:
            case "ASK":
                return self.data[1]
            case "BID":
                return self.data[3]
            case _:
                # Default is the average volume
                return (self.data[1] + self.data[3]) / 2.0


@dataclass(slots=True)
class Trade(PriceItem):
    trade_price: float
    trade_volume: float

    def price(self, price_type: str = "DEFAULT") -> float:
        return self.trade_price

    def volume(self, volume_type: str = "DEFAULT") -> float:
        return self.trade_volume


@dataclass(slots=True)
class Candle(PriceItem):
    ohlcv: array
    frequency: str = ""  # f.e 1s , 15m, 4h, 1d

    @classmethod
    def from_adj_close(cls, symbol, ohlcva: array, frequency=""):
        adj = ohlcva[5] / ohlcva[3]
        ohlcv = array("f", [ohlcva[0] * adj, ohlcva[1] * adj, ohlcva[2] * adj, ohlcva[5], ohlcva[4] / adj])
        return cls(symbol, ohlcv, frequency)

    def price(self, price_type: str = "DEFAULT") -> float:
        match price_type:
            case "OPEN":
                return self.ohlcv[0]
            case "HIGH":
                return self.ohlcv[1]
            case "LOW":
                return self.ohlcv[2]
            case _:
                return self.ohlcv[3]

    @property
    def close(self):
        return self.ohlcv[3]

    def volume(self, volume_type: str = "DEFAULT") -> float:
        return self.ohlcv[4]


class Event:
    """
    An event represents zero of items of information happening at a certain moment in time.
    An item can contain any type of information, but a common use-case are price-items like candles.
    Time is always a datetime object with UTC timezone.
    """

    def __init__(self, time: datetime, items: list[Any]):
        assert time.tzname() == "UTC", time.tzname()
        self.time = time
        self.items = items

    @staticmethod
    def empty(time=None):
        """Return a new empty event"""

        time = time or datetime.now(timezone.utc)
        return Event(time, [])

    def is_empty(self) -> bool:
        """return True if this is an empty event without any items, False otherwise"""
        return len(self.items) == 0

    @cached_property
    def price_items(self) -> dict[str, PriceItem]:
        """Returns the price-items in this event for each symbol.

        The first time this method is invoked, the result is calculated and cached.
        """
        return {item.symbol: item for item in self.items if isinstance(item, PriceItem)}

    def get_prices(self, price_type: str = "DEFAULT") -> dict[str, float]:
        """Return all the prices of a certain price_type"""
        return {k: v.price(price_type) for k, v in self.price_items.items()}

    def get_price(self, symbol: str, price_type: str = "DEFAULT") -> float | None:
        """Return the price for the symbol, or None if not found."""

        if item := self.price_items.get(symbol):
            return item.price(price_type)

    def __repr__(self) -> str:
        return f"Event(time={self.time} item={len(self.items)})"
