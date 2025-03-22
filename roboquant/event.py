from abc import abstractmethod
from array import array
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any

from roboquant.asset import Asset


@dataclass(slots=True)
class PriceItem:
    """Baseclass for the different types of prices related to an asset, like trades or quotes.

    Attributes:
        asset (Asset): The underlying asset for this price-item.
    """

    asset: Asset
    """The underlying asset for this price-item"""

    @abstractmethod
    def price(self, price_type: str = "DEFAULT") -> float:
        """Returns the price for the provided price_type. If a type is unknown, returns the `DEFAULT` price.

        Args:
            price_type (str): The type of price to return. A price-type is by convention always a capatalized string.
            For example, `OPEN`, `CLOSE`, `HIGH`, `LOW`, `BID` or `ASK.

        Returns:
            float: The price for the provided price_type. If the type is unknown, returns the `DEFAULT` price.
        """

    @abstractmethod
    def volume(self, volume_type: str = "DEFAULT") -> float:
        """Return the volume of the price-item. If a type is unknown, returns the `DEFAULT` volume.

        Args:
            volume_type (str): The type of volume to return. For example, `BID` or `ASK`.

        Returns:
            float: The volume for the provided volume_type. If the type is unknown, returns the `DEFAULT` volume.
        """


@dataclass(slots=True)
class Quote(PriceItem):
    """Quote price of an asset, containing ASK and BID prices.

    Attributes:
        data (array): An array containing [ask-price, ask-volume, bid-price, bid-volume].
    """

    data: array  # [ask-price, ask-volume, bid-price, bid-volume]

    def price(self, price_type: str = "MID") -> float:
        """Return the price, the default being the mid-point price.

        Args:
            price_type (str): The type of price to return. For example, `ASK` or `BID`.

        Returns:
            float: The price for the provided price_type. If the type is unknown, returns the mid-point price.
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
        """Return the ask price.

        Returns:
            float: The ask price.
        """
        return self.data[0]

    @property
    def bid_price(self) -> float:
        """Return the bid price.

        Returns:
            float: The bid price.
        """
        return self.data[2]

    @property
    def ask_volume(self) -> float:
        """Return the ask volume.

        Returns:
            float: The ask volume.
        """
        return self.data[1]

    @property
    def bid_volume(self) -> float:
        """Return the bid volume.

        Returns:
            float: The bid volume.
        """
        return self.data[3]

    @property
    def spread(self) -> float:
        """Return the spread between the ask and bid price.

        Returns:
            float: The spread between the ask and bid price.
        """
        return self.data[0] - self.data[2]

    @property
    def midpoint_price(self) -> float:
        """Return the mid-point price.

        Returns:
            float: The mid-point price.
        """
        return (self.data[0] + self.data[2]) / 2.0

    def volume(self, volume_type: str = "MID") -> float:
        """Return the volume, the default being the MID volume (average of BID and ASK volume).

        Args:
            volume_type (str): The type of volume to return. For example, `ASK` or `BID`.

        Returns:
            float: The volume for the provided volume_type. If the type is unknown, returns the MID volume.
        """

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

    Attributes:
        trade_price (float): The price of the trade.
        trade_volume (float): The volume of the trade.
    """

    trade_price: float
    """The price of the trade"""

    trade_volume: float
    """The volume of the trade"""

    def price(self, price_type: str = "DEFAULT") -> float:
        """Return the price of the trade.

        Args:
            price_type (str): The type of price to return. Default is `DEFAULT`.

        Returns:
            float: The price of the trade.
        """
        return self.trade_price

    def volume(self, volume_type: str = "DEFAULT") -> float:
        """Return the volume of the trade.

        Args:
            volume_type (str): The type of volume to return. Default is `DEFAULT`.

        Returns:
            float: The volume of the trade.
        """
        return self.trade_volume


@dataclass(slots=True)
class Bar(PriceItem):
    """Represents a bar (a.k.a. candlestick) with open-, high-, low-, close-price and volume data.

    Attributes:
        ohlcv (array): The open, high, low, close and volume data of the bar stored in an array of floats.
        frequency (str): The frequency of the bar, for example, 1s, 15m, 4h, 1d.
    """

    ohlcv: array  # [open, high, low, close, volume]
    """The open, high, low, close and volume data of the bar stored in an array of floats"""

    frequency: str = ""  # f.e 1s , 15m, 4h, 1d
    """The frequency of the bar, for example, 1s, 15m, 4h, 1d"""

    @classmethod
    def from_adj_close(cls, asset: Asset, ohlcv: array, adj_close: float, frequency=""):
        """Create a Bar instance from adjusted close price.

        Args:
            asset (Asset): The underlying asset.
            ohlcv (array): The open, high, low, close, and volume data.
            adj_close (float): The adjusted close price.
            frequency (str, optional): The frequency of the bar. Defaults to "".

        Returns:
            Bar: A new Bar instance with adjusted close price.
        """
        adj = adj_close / ohlcv[3]
        ohlcv = array("f", [ohlcv[0] * adj, ohlcv[1] * adj, ohlcv[2] * adj, adj_close, ohlcv[4] / adj])
        return cls(asset, ohlcv, frequency)

    def price(self, price_type: str = "CLOSE") -> float:
        """Return the price for the bar, default being the CLOSE price.

        Args:
            price_type (str): The type of price to return. For example, `OPEN`, `HIGH`, or `LOW`.

        Returns:
            float: The price for the provided price_type. If the type is unknown, returns the CLOSE price.
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
        """Return the volume of the bar.

        Args:
            volume_type (str): The type of volume to return. Default is `DEFAULT`.

        Returns:
            float: The volume of the bar.
        """
        return self.ohlcv[4]


class Event:
    """
    The `Event` class represents a snapshot of information occurring at a specific moment in time.
    It is designed to encapsulate a collection of items, such as market data or other relevant information,
    that are associated with a particular timestamp.
    Events often represent market activities like trades, quotes, or bars. However, it is flexible enough to handle
    other types of data, such as fundamental metrics or social media signals.

    Args:
        time (datetime): The timestamp of the event, which must be in UTC timezone. This ensures
                         consistency when dealing with events across different systems.
        items (list[Any]): A list of items associated with the event. These items can represent
                           various types of information, such as `PriceItem` instances (e.g., `Quote`,
                           `Trade`, or `Bar`) or other custom data types.
    Methods:
        __init__(dt: datetime, items: list[Any]):
            Initializes an `Event` instance with a specific timestamp and a list of associated items.
        empty(dt: datetime) -> Event:
            Creates and returns an empty `Event` instance with the specified timestamp.
        is_empty() -> bool:
            Checks whether the event contains any items. Returns `True` if the event is empty,
            otherwise `False`.
        price_items() -> dict[Asset, PriceItem]:
            A cached property that returns a dictionary mapping each asset to its corresponding
            price item. This is useful for quickly accessing price-related data.
        get_prices(price_type: str = "DEFAULT") -> dict[Asset, float]:
            Retrieves the prices of all assets in the event for a specified price type. Returns a
            dictionary mapping each asset to its price.
        get_price(asset: Asset, price_type: str = "DEFAULT") -> float | None:
            Retrieves the price of a specific asset for a given price type. Returns the price as a
            float, or `None` if the asset is not found.
        get_volume(asset: Asset, volume_type: str = "DEFAULT") -> float | None:
            Retrieves the volume of a specific asset for a given volume type. Returns the volume as
            a float, or `None` if the asset is not found.
        __repr__() -> str:
            Returns a string representation of the `Event` instance, including its timestamp and
            the number of items it contains.
    """

    def __init__(self, dt: datetime, items: list[Any]):
        """Initialize an Event instance.

        Args:
            dt (datetime): The datetime of the event. Must be in UTC timezone.
            items (list[Any]): A list of items associated with the event.
        """
        assert dt.tzname() == "UTC", "event with non UTC timezone"
        self.time: datetime = dt
        self.items: list[Any] = items

    @staticmethod
    def empty(dt: datetime):
        """Return a new empty event at the provided time.

        Args:
            dt (datetime): The datetime of the event. Must be in UTC timezone.

        Returns:
            Event: A new empty event.
        """
        return Event(dt, [])

    def is_empty(self) -> bool:
        """Return True if this is an empty event without any items, False otherwise.

        Returns:
            bool: True if the event is empty, False otherwise.
        """
        return len(self.items) == 0

    @cached_property
    def price_items(self) -> dict[Asset, PriceItem]:
        """Returns the price-items in this event for each asset.

        Returns:
            dict[Asset, PriceItem]: A dictionary mapping each asset to its corresponding price-item.

        Note:
            The first time this method is invoked, the result is calculated and cached.
        """
        return {item.asset: item for item in self.items if isinstance(item, PriceItem)}

    def get_prices(self, price_type: str = "DEFAULT") -> dict[Asset, float]:
        """Return all the prices of a certain price_type.

        Args:
            price_type (str): The type of price to return. Default is `DEFAULT`.

        Returns:
            dict[Asset, float]: A dictionary mapping each asset to its corresponding price.
        """
        return {k: v.price(price_type) for k, v in self.price_items.items()}

    def get_price(self, asset: Asset, price_type: str = "DEFAULT") -> float | None:
        """Return the price for the asset, or None if not found.

        Args:
            asset (Asset): The asset for which to return the price.
            price_type (str): The type of price to return. Default is `DEFAULT`.

        Returns:
            float | None: The price for the asset, or None if not found.
        """
        if item := self.price_items.get(asset):
            return item.price(price_type)
        return None

    def get_volume(self, asset: Asset, volume_type: str = "DEFAULT") -> float | None:
        """Return the volume for the asset, or None if not found.

        Args:
            asset (Asset): The asset for which to return the volume.
            volume_type (str): The type of volume to return. Default is `DEFAULT`.

        Returns:
            float | None: The volume for the asset, or None if not found.
        """
        if item := self.price_items.get(asset):
            return item.volume(volume_type)
        return None

    def __repr__(self) -> str:
        """Return a string representation of the event.

        Returns:
            str: A string representation of the event.
        """
        return f"Event(time={self.time} items={len(self.items)})"
