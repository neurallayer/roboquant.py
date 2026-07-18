import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Type

from roboquant.monetary import USD, Amount, Currency

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Asset(ABC):
    """Abstract base class for all types of assets, ranging from stocks to cryptocurrencies.
    Every asset has always at least a `symbol` and `currency` defined. Assets are immutable.

    The combination of the class, symbol, and currency has to be unique for each asset. If that is not the case, the symbol
    could be extended with some additional information to make it unique. For example, for stocks,
    the exchange could be added to the symbol.
    """

    symbol: str
    """The symbol name of the asset, for example, AAPL"""

    currency: Currency = USD
    """The currency of the asset, default is `USD`"""

    def contract_value(self, size: Decimal, price: float) -> float:
        """Return the total contract value given the provided size and price.
        The default implementation simply multiplies the size with the price, but for some asset types, like options,
        this can be overridden to include a contract size multiplier.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            float: The total contract value.
        """
        return float(size) * price

    def contract_amount(self, size: Decimal, price: float) -> Amount:
        """Return the total contract amount given the provided size and price.
        The returned amount is denoted in the currency of the asset.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            Amount: The total contract amount.
        """
        value = self.contract_value(size, price)
        return Amount(self.currency, value)

    def __eq__(self, value: object) -> bool:
        """Check if two assets are equal based on their class, symbol and currency.

        Args:
            value (object): The other asset to compare with.

        Returns:
            bool: True if the assets are equal, False otherwise.
        """
        if value is self:
            return True

        if isinstance(value, self.__class__):
            return self.symbol == value.symbol and self.currency == value.currency

        return False

    def __hash__(self):
        """Return the hash of the asset based on its symbol.

        Returns:
            int: The hash value of the asset.
        """
        return hash(self.symbol)

    def asset_class(self) -> str:
        """Return the class name of the asset, the default implementation returns the class name of the instance.

        Returns:
            str: The class name of the asset.
        """
        return self.__class__.__name__

    @abstractmethod
    def serialize(self) -> str:
        """Serialize the asset to a string representation that can be used to reconstruct the asset later on.

        Returns:
            str: The serialized string representation of the asset.
        """
        ...

    @staticmethod
    @abstractmethod
    def deserialize(value: str) -> "Asset":
        """Deserialize a string value to an asset.
        This method should be able to deserialize the string that was created using the `serialize` method.

        Args:
            value (str): The serialized string representation of the asset.

        Returns:
            Asset: The deserialized asset.
        """
        ...


@dataclass(frozen=True, slots=True)
class Stock(Asset):
    """Tradable stock or equity asset.

    A stock represents ownership in a publicly traded company. The `symbol`
    should identify the listing unambiguously within the feed or broker
    being used, for example `AAPL` or `NASDAQ:AAPL` when an exchange qualifier
    is needed. Stocks default to being denominated in `USD`, but a different
    `Currency` can be provided for non-US listings.
    """

    def serialize(self):
        """Serialize the stock asset to a string representation.

        Returns:
            str: The serialized string representation of the stock asset.
        """
        return f"Stock:{self.symbol}:{self.currency}"

    @staticmethod
    def deserialize(value: str) -> "Stock":
        """Deserialize a string value to a stock asset.

        Args:
            value (str): The serialized string representation of the stock asset.

        Returns:
            Stock: The deserialized stock asset.
        """
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Stock"
        return Stock(symbol, Currency(currency))


@dataclass(frozen=True, slots=True)
class Crypto(Asset):
    """Tradable cryptocurrency asset or crypto pair.

    A crypto asset is typically expressed as a base/quote pair, such as
    `BTC/USDT` or `ETH/EUR`. The full pair is kept as the `symbol`, while the
    quote part identifies the asset currency. Use `from_symbol` for common pair
    formats so the currency can be derived from the symbol automatically.
    """

    @staticmethod
    def from_symbol(symbol: str):
        """Create a Crypto asset from a symbol string. It will automatically extract the quote currency from the symbol,
        which is assumed to be the last part of the symbol.

        Args:
            symbol (str): The symbol string of the crypto asset.

        Returns:
            Crypto: The created crypto asset.
        """
        currency = re.split(r"[^a-zA-Z0-9\s]", symbol)[-1]
        return Crypto(symbol, Currency(currency))

    def serialize(self):
        """Serialize the crypto asset to a string representation.

        Returns:
            str: The serialized string representation of the crypto asset.
        """
        return f"Crypto:{self.symbol}:{self.currency}"

    @staticmethod
    def deserialize(value: str) -> "Crypto":
        """Deserialize a string value to a crypto asset.

        Args:
            value (str): The serialized string representation of the crypto asset.

        Returns:
            Crypto: The deserialized crypto asset.
        """
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Crypto"
        return Crypto(symbol, Currency(currency))


@dataclass(frozen=True, slots=True)
class Forex(Asset):
    """Foreign exchange currency-pair asset.

    A forex asset represents one currency traded against another, usually
    written as a base/quote pair such as `EUR/USD` or `GBP/JPY`. The complete
    pair is stored as the `symbol`, and the quote currency is used as the
    asset currency. Use `from_symbol` for standard pair notation so the quote
    currency can be inferred automatically.
    """

    @staticmethod
    def from_symbol(symbol: str) -> "Forex":
        """Create a Forex asset from a symbol string. The last part of the symbol is assumed to be the
        currency.

        Args:
            symbol (str): The symbol string of the forex asset.
        Returns:
            Forex: The created forex asset.
        """
        currency = re.split(r"[^a-zA-Z0-9\s]", symbol)[-1]
        return Forex(symbol, Currency(currency))

    def serialize(self):
        """Serialize the crypto asset to a string representation.

        Returns:
            str: The serialized string representation of the crypto asset.
        """
        return f"Forex:{self.symbol}:{self.currency}"

    @staticmethod
    def deserialize(value: str) -> "Forex":
        """Deserialize a string value to a crypto asset.

        Args:
            value (str): The serialized string representation of the crypto asset.

        Returns:
            Forex: The deserialized crypto asset.
        """
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Forex"
        return Forex(symbol, Currency(currency))


@dataclass(frozen=True, slots=True)
class Option(Asset):
    """Option contract asset with a standard contract multiplier.

    An option represents the right to buy or sell an underlying asset at a
    specified strike and expiry. The `symbol` should identify the contract
    unambiguously, for example using an OCC-style option symbol. Option prices
    are quoted per underlying unit, so this class multiplies contract value by
    `100` to model the standard US equity option contract size.
    """

    def contract_value(self, size: Decimal, price: float) -> float:
        """Contract value for this option type is the `size` times the `price` times `100`.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            float: The total contract value.
        """
        return float(size) * price * 100.0

    @staticmethod
    def deserialize(value: str) -> "Option":
        """Deserialize a string value to an option asset.

        Args:
            value (str): The serialized string representation of the option asset.

        Returns:
            Option: The deserialized option asset.
        """
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Option"
        return Option(symbol, Currency(currency))

    def serialize(self):
        """Serialize the option asset to a string representation.

        Returns:
            str: The serialized string representation of the option asset.
        """
        return f"Option:{self.symbol}:{self.currency}"


# Keep the registered asset classes in a dictionary so they can be deserialized later on
__asset_classes: dict[str, Type[Asset]] = {}
__cache: dict[str, Asset] = {}


def register_asset_class(clazz: Type[Asset]):
    """Register an asset class so it can be deserialized later on.

    Args:
        clazz (Type[Asset]): The asset class to register.
    """
    __asset_classes[clazz.__name__] = clazz
    logging.info("registered asset class %s", clazz.__name__)


def deserialize_to_asset(value: str) -> Asset:
    """Based on the provided string value, deserialize it to the correct asset. The asset class needs to be registered
    first using the `register_asset_class` method.

    Under the hood is uses caching to improve performance when repeatedly deserializing the same asset strings.

    Args:
        value (str): The serialized string representation of the asset.

    Returns:
        Asset: The deserialized asset.
    """
    asset = __cache.get(value)
    if not asset:
        asset_class, _ = value.split(":", maxsplit=1)
        deserializer = __asset_classes[asset_class].deserialize
        asset = deserializer(value)
        __cache[value] = asset
    return asset


# Register the default included asset classes
register_asset_class(Stock)
register_asset_class(Option)
register_asset_class(Crypto)
register_asset_class(Forex)
