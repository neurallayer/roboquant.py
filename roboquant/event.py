from abc import abstractmethod
from array import array
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any

from roboquant.asset import Asset


@dataclass(slots=True)
class PriceItem:
    """Baseclass for the different types of prices related to an asset, like trades or quotes
    """

    asset: Asset
    """the underlying asset for this price-item"""

    @abstractmethod
    def price(self, price_type: str = "DEFAULT") -> float:
        """Returns the price for the provided price_type.
        A price_type, for example, is `OPEN` or `CLOSE`.

        All price-items are expected to return the `DEFAULT` price if the type is unknown.
        """

    @abstractmethod
    def volume(self, volume_type: str = "DEFAULT") -> float:
        """Return the volume of the price-item.
        Some price-items have multiple volumes, like the BID and ASK volume.
        """


@dataclass(slots=True)
class Quote(PriceItem):
    """Quote price of an asset, containing ASK and BID prices"""

    data: array  # [ask-price, ask-volume, bid-price, bid-volume]

    def price(self, price_type: str = "MID") -> float:
        """Return the price, the default being the mid-point price.
        Alternatively, you can request the ASK or BID price.
        """

        match price_type:
            case "ASK":
                return self.data[0]
            case "BID":
                return self.data[2]
            case _:
                # Default is the mid-point price
                return (self.data[0] + self.data[2]) / 2.0

    @property
    def ask_price(self) -> float:
        """Return the ask price"""
        return self.data[0]

    @property
    def bid_price(self) -> float:
        """Return the bid price"""
        return self.data[2]

    @property
    def ask_volume(self) -> float:
        """Return the ask volume"""
        return self.data[1]

    @property
    def bid_volume(self) -> float:
        """Return the bid volume"""
        return self.data[3]

    @property
    def spread(self) -> float:
        """Return the spread between the ask and bid price"""
        return self.data[0] - self.data[2]

    @property
    def midpoint_price(self) -> float:
        """Return the mid-point price"""
        return (self.data[0] + self.data[2]) / 2.0

    def volume(self, volume_type: str = "MID") -> float:
        """Return the volume, the default being the MID volume (average of BID and ASK volume).
        Alternatively, you can request the ASK or BID volume."""

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
    """Holds a single price and optional the volume.
    Often this reflects an actual trade, but it can also be used in different scenarios.
    """
    trade_price: float
    """The price of the trade"""

    trade_volume: float
    """The volume of the trade"""

    def price(self, price_type: str = "DEFAULT") -> float:
        """Return the price of the trade"""
        return self.trade_price

    def volume(self, volume_type: str = "DEFAULT") -> float:
        """Return the volume of the trade"""
        return self.trade_volume


@dataclass(slots=True)
class Bar(PriceItem):
    """Represents a bar (a.k.a. candlestick) with open-, high-, low-, close-price and volume data.
    """

    ohlcv: array  # [open, high, low, close, volume]
    """The open, high, low, close and volume data of the bar stored in an array of floats"""

    frequency: str = ""  # f.e 1s , 15m, 4h, 1d
    """The frequency of the bar, for example, 1s, 15m, 4h, 1d"""

    @classmethod
    def from_adj_close(cls, asset: Asset, ohlcv: array, adj_close: float, frequency=""):
        adj = adj_close / ohlcv[3]
        ohlcv = array("f", [ohlcv[0] * adj, ohlcv[1] * adj, ohlcv[2] * adj, adj_close, ohlcv[4] / adj])
        return cls(asset, ohlcv, frequency)

    def price(self, price_type: str = "CLOSE") -> float:
        """Return the price for the bar, default being the CLOSE price.
        Alternatively, you can request the OPEN, HIGH or LOW price.
        """
        match price_type:
            case "DEFAULT":
                return self.ohlcv[3]
            case "OPEN":
                return self.ohlcv[0]
            case "HIGH":
                return self.ohlcv[1]
            case "LOW":
                return self.ohlcv[2]
            case _:
                return self.ohlcv[3]

    def volume(self, volume_type: str = "DEFAULT") -> float:
        return self.ohlcv[4]


class Event:
    """
    An event contains zero or more items of information happening at the same moment in time.

    - `Event.time` is a datetime object with the timezone set at UTC.
    - An item can be any type of object. The most common use-cases are of the type `PriceItem`,
      like `Quote`, `Trade` or `Bar`. But items could also represent other types of information like
      fundamental data or social-media posts.
    """

    def __init__(self, dt: datetime, items: list[Any]):
        assert dt.tzname() == "UTC", "event with non UTC timezone"
        self.time: datetime = dt
        self.items: list[Any] = items

    @staticmethod
    def empty(dt: datetime):
        """Return a new empty event at the provided time"""
        return Event(dt, [])

    def is_empty(self) -> bool:
        """return True if this is an empty event without any items, False otherwise"""
        return len(self.items) == 0

    @cached_property
    def price_items(self) -> dict[Asset, PriceItem]:
        """Returns the price-items in this event for each asset.

        The first time this method is invoked, the result is calculated and cached.
        """
        return {item.asset: item for item in self.items if isinstance(item, PriceItem)}

    def get_prices(self, price_type: str = "DEFAULT") -> dict[Asset, float]:
        """Return all the prices of a certain price_type"""
        return {k: v.price(price_type) for k, v in self.price_items.items()}

    def get_price(self, asset: Asset, price_type: str = "DEFAULT") -> float | None:
        """Return the price for the asset, or None if not found."""

        if item := self.price_items.get(asset):
            return item.price(price_type)
        return None

    def get_volume(self, asset: Asset, volume_type: str = "DEFAULT") -> float | None:
        """Return the volume for the asset, or None if not found."""

        if item := self.price_items.get(asset):
            return item.volume(volume_type)
        return None

    def __repr__(self) -> str:
        return f"Event(time={self.time} items={len(self.items)})"
