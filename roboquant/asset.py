from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Type

from roboquant.monetary import Amount, Currency, USD


@dataclass(frozen=True, slots=True)
class Asset(ABC):
    """Abstract baseclass for all types of assets, ranging from stocks to cryptocurrencies.
    Every asset has always at least a `symbol` and `currency` defined. Assets are immutable.

    The combination of class, symbol and currency should be unique for each asset. If that is not the case, the symbol
    could be extended with some additional information to make it unique. For example, for stocks,
    the exchange could be added to the symbol.
    """

    symbol: str
    """The symbol name of the asset, for example, AAPL"""

    currency: Currency = USD
    """The currency of the asset, default is USD"""

    def contract_value(self, size: Decimal, price: float) -> float:
        """return the total contract value given the provided size and price.
        The default implementation for the contract value is the `size` times the `price`.
        """
        return float(size) * price

    def contract_amount(self, size: Decimal, price: float) -> Amount:
        """return the total contract amount given the provided size and price
        The returned amount is denoted in the currency of the asset.
        """
        value = self.contract_value(size, price)
        return Amount(self.currency, value)

    def __eq__(self, value: object) -> bool:
        if value is self:
            return True

        if isinstance(value, self.__class__):
            return self.symbol == value.symbol and self.currency == value.currency

        return False

    def __hash__(self):
        return hash(self.symbol)

    @abstractmethod
    def serialize(self) -> str:
        """Serialize the asset to a string representation that can be used to reconstruct the asset later on.
        The first part of the string should be the `class.__name__`, followed by a semicolon.
        For example, `Stock:AAPL:USD`
        """
        ...

    @staticmethod
    @abstractmethod
    def deserialize(value: str) -> "Asset":
        """Deserialize a string value to an asset.
        This method should be able to deserialize the string that was created using the `serialize` method"""
        ...


@dataclass(frozen=True, slots=True)
class Stock(Asset):
    """Stock (or equity) asset"""

    @staticmethod
    def asset_class() -> str:
        return "Stock"

    def serialize(self):
        return f"Stock:{self.symbol}:{self.currency}"

    @staticmethod
    def deserialize(value: str) -> "Stock":
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Stock"
        return Stock(symbol, Currency(currency))


@dataclass(frozen=True, slots=True)
class Crypto(Asset):
    """Crypto-currency asset"""

    @staticmethod
    def from_symbol(symbol: str, sep: str="/"):
        currency = symbol.split(sep)[-1]
        return Crypto(symbol, Currency(currency))

    def serialize(self):
        return f"Crypto:{self.symbol}:{self.currency}"

    @staticmethod
    def deserialize(value: str) -> "Crypto":
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Crypto"
        return Crypto(symbol, Currency(currency))


@dataclass(frozen=True, slots=True)
class Option(Asset):
    """Option Contract asset that has uses a contract size of 100 to calculate the contract value"""

    def contract_value(self, size: Decimal, price: float) -> float:
        """Contract value for this option type is the `size` times the `price` times `100`"""
        return float(size) * price * 100.0

    @staticmethod
    def deserialize(value: str) -> "Option":
        asset_class, symbol, currency = value.split(":")
        assert asset_class == "Option"
        return Option(symbol, Currency(currency))

    def serialize(self):
        return f"Option:{self.symbol}:{self.currency}"

# Keep the registered asset classes in a dictionary so they can be deserialized later on
__asset_classes: dict[str, Type[Asset]] = {}
__cache: dict[str, Asset] = {}

def register_asset_class(clazz: Type[Asset]):
    """Register an asset class so it can be deserialized later on"""
    __asset_classes[clazz.__name__] = clazz

def deserialize_to_asset(value: str) -> Asset:
    """Based on the provided string value, deserialize it to the correct asset. The asset class needs to be registered
    first using the `register_asset_class` method.
    """
    asset = __cache.get(value)
    if not asset:
        asset_class, _ = value.split(":", maxsplit=1)
        deserializer = __asset_classes[asset_class].deserialize
        asset = deserializer(value)
        __cache[value] = asset
    return asset


# Register the default inlcuded asset classes
register_asset_class(Stock)
register_asset_class(Option)
register_asset_class(Crypto)


