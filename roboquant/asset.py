from abc import ABC
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from roboquant.wallet import Amount


@dataclass(frozen=True, slots=True)
class Asset(ABC):

    symbol: str
    currency: str = "USD"

    __cache: ClassVar[dict[str, "Asset"]] = {}

    def contract_value(self, size: Decimal, price: float) -> float:
        return float(size) * price

    def contract_amount(self, size: Decimal, price: float) -> Amount:
        value = self.contract_value(size, price)
        return Amount(self.currency, value)

    def type(self):
        return type(self).__name__

    def __eq__(self, value: object) -> bool:
        if value is self:
            return True

        if isinstance(value, self.__class__):
            return self.symbol == value.symbol and self.currency == value.currency

        return False

    def __hash__(self):
        return hash(self.symbol)

    def serialize(self):
        return self.type() + ":" + self.symbol + ":" + self.currency

    @staticmethod
    def deserialize(value: str) -> "Asset":
        asset = Asset.__cache.get(value)
        if not asset:
            asset_type, other = value.split(":", maxsplit=1)
            d = _asset_deserializer_registry[asset_type]
            asset = d(other)
            Asset.__cache[value] = asset
        return asset


@dataclass(frozen=True, slots=True)
class Stock(Asset):

    def serialize(self):
        return "Stock" + ":" + self.symbol + ":" + self.currency


@dataclass(frozen=True, slots=True)
class Crypto(Asset):

    symbol: str  # type: ignore
    currency: str  # type: ignore

    @staticmethod
    def from_symbol(symbol: str, sep="/"):
        currency = symbol.split(sep)[-1]
        return Crypto(symbol, currency)


@dataclass(frozen=True, slots=True)
class Option(Asset):
    multiplier = 100

    def contract_value(self, size: Decimal, price: float) -> float:
        return float(size) * price * self.multiplier


def __default_deserializer(clazz):

    __cache: dict[str, Asset] = {}

    def _deserialize(value: str) -> Asset:
        asset = __cache.get(value)
        if not asset:
            symbol, currency = value.split(":")
            asset = clazz(symbol, currency)
            __cache[value] = asset
        return asset

    return _deserialize


_asset_deserializer_registry = {
    "Stock": __default_deserializer(Stock),
    "Option": __default_deserializer(Option),
    "Crypto": __default_deserializer(Crypto),
}
