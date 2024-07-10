from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar


class CurrencyConverter:

    @abstractmethod
    def convert(self, from_currency: str, to_currency: str, value: float, time: datetime | None) -> float:
        pass


class NoConversion(CurrencyConverter):

    def convert(self, from_currency: str, to_currency: str, value: float, time: datetime | None) -> float:
        raise NotImplementedError("The default NoConversion doesn't support conversions")


class One2OneConversion(CurrencyConverter):

    def convert(self, from_currency: str, to_currency: str, value: float, time: datetime | None) -> float:
        return value


@dataclass(frozen=True, slots=True)
class Amount:

    currency: str
    value: float
    converter: ClassVar[CurrencyConverter] = NoConversion()

    def items(self):
        return [(self.currency, self.value)]

    def amounts(self):
        return [self]

    def convert(self, currency: str, time: datetime | None = None) -> float:
        if currency == self.currency:
            return self.value
        if self.value == 0.0:
            return 0.0

        return Amount.converter.convert(self.currency, currency, self.value, time)

    def __repr__(self) -> str:
        return f"{self.value} {self.currency}"


class Wallet(defaultdict[str, float]):

    def __init__(self, *amounts: Amount):
        super().__init__(float)
        for amount in amounts:
            self[amount.currency] += amount.value

    def amounts(self):
        return [Amount(k, v) for k, v in self.items()]

    def __iadd__(self, other: "Amount | Wallet"):
        for k, v in other.items():
            self[k] += v
        return self

    def __isub__(self, other: "Amount | Wallet"):
        for k, v in other.items():
            self[k] -= v
        return self

    def __add__(self, other: "Amount | Wallet"):
        result = self.copy()
        for k, v in other.items():
            result[k] += v
        return result

    def __sub__(self, other: "Amount | Wallet"):
        result = self.copy()
        for k, v in other.items():
            result[k] -= v
        return result

    def copy(self) -> "Wallet":
        result = Wallet()
        result.update(self)
        return result

    def convert(self, currency: str, time: datetime | None = None) -> float:
        return sum(amount.convert(currency, time) for amount in self.amounts())

    def __repr__(self) -> str:
        return ", ".join([f"{a}" for a in self.amounts()])
