import io
import logging
import zipfile
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from csv import reader
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Any, Dict, List

import requests


logger = logging.getLogger(__name__)


class Currency(str):
    """Currency class represents a monetary currency and is s subclass of `str`"""

    def __rmatmul__(self, other: float | int):
        assert isinstance(other, (float, int))
        return Amount(self, other)


# Commonly used currencies
USD = Currency("USD")
EUR = Currency("EUR")
JPY = Currency("JPY")
GBP = Currency("GBP")
CHF = Currency("CHF")
INR = Currency("INR")
AUD = Currency("AUD")
CAD = Currency("CAD")
NZD = Currency("NZD")
CMY = Currency("CMY")
HKD = Currency("HKD")
BTC = Currency("BTC")
ETH = Currency("ETH")


class CurrencyConverter(ABC):
    """Abstract base class for currency converters"""

    @abstractmethod
    def convert(self, amount: "Amount", to_currency: Currency, time: datetime) -> float:
        """Convert the monetary amount into another currency at the provided time."""
        ...


class NoConversion(CurrencyConverter):
    """The default currency converter that doesn't convert between currencies"""

    def convert(self, amount: "Amount", to_currency: Currency, time: datetime) -> float:
        raise NotImplementedError("The default NoConversion doesn't support any conversions")


class ECBConversion(CurrencyConverter):
    """Currency that gets it exchange rates from the European Central Bank"""

    __file_name = Path.home() / ".roboquant" / "eurofxref-hist.csv"

    def __init__(self):
        self.rates: Dict[Currency, List[Any]] = {EUR: [(datetime.fromisoformat("2000-01-01T15:00:00+00:00"), 1.0)]}
        if not self.exists():
            self.download()
        self.parse()

    def download(self):
        if self.exists():
            logging.info("overwriting existing file")

        r = requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip", timeout=5)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            p = Path.home() / ".roboquant"
            z.extractall(p)

    def exists(self):
        self.__file_name.exists()

    def parse(self):
        with open(self.__file_name, "r", encoding="utf8") as csv_file:
            csv_reader = reader(csv_file)
            header = next(csv_reader)[1:]
            currencies = [Currency(e) for e in header if e]
            for e in header:
                c = Currency(e)
                self.rates[c] = []

            header_len = len(currencies)
            for row in csv_reader:
                d = datetime.fromisoformat(row[0] + "T15:00:00+00:00")
                for idx in range(header_len):
                    v = row[idx + 1]
                    if v and v != "N/A":
                        value = (d, float(v))
                        self.rates[currencies[idx]].append(value)

        for v in self.rates.values():
            v.reverse()

    def get_rate(self, currency: Currency, time: datetime) -> float:
        rates = self.rates[currency]
        idx = bisect_left(rates, time, key=lambda r: r[0])
        idx = min(idx, len(rates) - 1)
        return rates[idx][1]

    def convert(self, amount: "Amount", to_currency: Currency, time: datetime) -> float:
        return amount.value * self.get_rate(to_currency, time) / self.get_rate(amount.currency, time)


class StaticConversion(CurrencyConverter):
    """Currency converter that uses static rates to convert between different currencies.
    This converter doesn't take time into consideration.
    """

    def __init__(self, base_currency: Currency, rates: dict[Currency, float]):
        super().__init__()
        self.base_currency = base_currency
        self.rates = rates
        self.rates[base_currency] = 1.0

    def convert(self, amount: "Amount", to_currency: Currency, time: datetime) -> float:
        return amount.value * self.rates[to_currency] / self.rates[amount.currency]


class One2OneConversion(CurrencyConverter):
    """Currency converter that always converts 1 to 1 between currencies.
    So for example, 1 USD equals 1 EUR equals 1 GPB"""

    def convert(self, amount: "Amount", to_currency: str, time: datetime) -> float:
        return amount.value


@dataclass(frozen=True, slots=True)
class Amount:
    """A monetary value denoted in a single currency. Amounts are immutable"""

    currency: Currency
    value: float
    __converter: ClassVar[CurrencyConverter] = NoConversion()

    @staticmethod
    def register_converter(converter: CurrencyConverter):
        """Register a new currency converter to handle conversions between different currencies"""
        Amount.__converter = converter

    def items(self):
        return [(self.currency, self.value)]

    def amounts(self):
        return [self]

    def __add__(self, other: "Amount") -> "Wallet":
        """Add another amount to this amount.
        This will always return a wallet, even if both amounts have the same currency.
        """
        return Wallet(self, other)

    def convert(self, currency: Currency, time: datetime) -> float:
        """Convert this amount to another currency and return that value.
        If a conversion is required, it will invoke the registered `Amount.converter`.
        """
        if currency == self.currency:
            return self.value
        if self.value == 0.0:
            return 0.0

        return Amount.__converter.convert(self, currency, time)

    def __repr__(self) -> str:
        return f"{self.value:,.2f}@{self.currency}"


class Wallet(defaultdict[Currency, float]):
    """A wallet holds monetary values of different currencies"""

    def __init__(self, *amounts: Amount):
        super().__init__(float)
        for amount in amounts:
            self[amount.currency] += amount.value

    def amounts(self):
        """Return the amounts contained in this wallet"""
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
        result = self.deepcopy()
        for k, v in other.items():
            result[k] += v
        return result

    def __sub__(self, other: "Amount | Wallet"):
        result = self.deepcopy()
        for k, v in other.items():
            result[k] -= v
        return result

    def deepcopy(self) -> "Wallet":
        result = Wallet()
        result.update(self)
        return result

    def convert(self, currency: Currency, time: datetime) -> float:
        """convert all the amounts hold in this wallet to a single currency and return the value"""
        return sum(amount.convert(currency, time) for amount in self.amounts())

    def __repr__(self) -> str:
        return " + ".join([f"{a}" for a in self.amounts()])


if __name__ == "__main__":
    ECBConversion().download()
    Amount.register_converter(ECBConversion())
    amt = 100 @ EUR + 10_000 @ JPY
    dt = datetime.fromisoformat("2024-01-01T00:00:00Z")
    print(amt.convert(USD, dt))
