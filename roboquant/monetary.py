import io
import logging
import os
from time import time
import zipfile
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from csv import reader
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Any, Dict, List

import requests


logger = logging.getLogger(__name__)


class Currency(str):
    """Currency class represents a monetary currency and is a subclass of `str`.

    It is possible to create an `Amount` using a combination of a `number` and a `Currency`:
    ```
    amount1 = 100@USD
    amount2 = 200.50@EUR
    amount3 = USD(100)
    ```
    """

    def __rmatmul__(self, other: float | int):
        """Create a new `Amount` using this currency and the provided `other` value.

        Args:
            other (float | int): The monetary value.

        Returns:
            Amount: The resulting Amount object.

        Example:
        ```
        amount1 = 100@USD
        ```
        """
        assert isinstance(other, (float, int))
        return Amount(self, other)

    def __call__(self, other: float | int) -> "Amount":
        """Create a new `Amount` using this currency and the provided value.

        Args:
            other (float | int): The monetary value.

        Returns:
            Amount: The resulting Amount object.

        Example:
        ```
        amount = USD(100)
        ```
        """
        return Amount(self, other)


# Commonly used currencies
USD = Currency("USD")
"""US Dollar"""

EUR = Currency("EUR")
"""Euro"""

JPY = Currency("JPY")
"""Japanese Yen"""

GBP = Currency("GBP")
"""British Pound"""

CHF = Currency("CHF")
"""Swiss Franc"""

INR = Currency("INR")
"""Indian Rupee"""

AUD = Currency("AUD")
"""Australian Dollar"""

CAD = Currency("CAD")
"""Canadian Dollar"""

NZD = Currency("NZD")
"""New Zealand Dollar"""

CMY = Currency("CMY")
"""Chinese Yuan"""

HKD = Currency("HKD")
"""Hong Kong Dollar"""

BTC = Currency("BTC")
"""Bitcoin"""

ETH = Currency("ETH")
"""Ethereum"""

USDT = Currency("USDT")
"""Tether USD"""

USDC = Currency("USDC")
"""USD Coin"""


class CurrencyConverter(ABC):
    """Abstract base class for currency converters. They are used to convert monetary amounts from one currency to another."""

    @abstractmethod
    def convert(self, amount: "Amount", to_currency: Currency, dt: datetime) -> float:
        """Convert the monetary amount into another currency at the provided time.

        Args:
            amount (Amount): The amount to convert.
            to_currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        ...

    def register(self):
        """Register this converter to be used for all conversions between amounts."""
        Amount.register_converter(self)


class NoConversion(CurrencyConverter):
    """The default currency converter that doesn't convert between currencies. If you don't trade in different currencies, this
    will be sufficient."""

    def convert(self, amount: "Amount", to_currency: Currency, dt: datetime) -> float:
        """Raise an error as no conversion is supported."""
        raise NotImplementedError("The default NoConversion doesn't support any conversions")


class ECBConversion(CurrencyConverter):
    """CurrencyConverter that retrieves its exchange rates from the ECB (European Central Bank).
    These exchange rates are based on the Euro and are updated daily by the ECB. They don't contain
    any historical data from before the Euro was introduced in 1999.
    """

    __file_name = Path.home() / ".roboquant" / "eurofxref-hist.csv"

    def __init__(self, force_download: bool=False):
        """Initialize the ECBConversion.

        Args:
            force_download (bool): Force download of the latest exchange rates if True.
        """
        self._rates: Dict[Currency, List[Any]] = {}
        if force_download or not self.up_to_date():
            self._download()
        self._parse()

    def _download(self):
        """Download the latest ECB exchange rates."""
        logging.info("downloading the latest ECB exchange rates")
        r = requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip", timeout=10)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            p = Path.home() / ".roboquant"
            z.extractall(p)

    def up_to_date(self):
        """Check if there is already a recently (< 12 hours) downloaded file.

        Returns:
            bool: True if the file is up to date, False otherwise.
        """
        if not self.__file_name.exists():
            return False
        diff = time() - os.path.getctime(self.__file_name)
        return diff < 12.0 * 3600.0

    @property
    def currencies(self) -> set[Currency]:
        """Return the set of supported currencies.

        Returns:
            set[Currency]: The set of supported currencies.
        """
        return set(self._rates.keys())

    def _parse(self):
        """Parse the downloaded exchange rates file."""
        self._rates = {EUR: [(datetime.fromisoformat("2000-01-01T15:00:00+00:00"), 1.0)]}
        with open(self.__file_name, "r", encoding="utf8") as csv_file:
            csv_reader = reader(csv_file)
            header = next(csv_reader)[1:]
            currencies = [Currency(e) for e in header if e]
            for e in header:
                c = Currency(e)
                self._rates[c] = []

            header_len = len(currencies)
            for row in csv_reader:
                d = datetime.fromisoformat(row[0] + "T15:00:00+00:00")
                for idx in range(header_len):
                    v = row[idx + 1]
                    if v and v != "N/A":
                        value = (d, float(v))
                        self._rates[currencies[idx]].append(value)

        for v in self._rates.values():
            v.reverse()

    def _get_rate(self, currency: Currency, dt: datetime) -> float:
        """Get the exchange rate for a currency at a specific datetime.

        Args:
            currency (Currency): The currency to get the rate for.
            dt (datetime): The datetime for the rate.

        Returns:
            float: The exchange rate.
        """
        rates = self._rates[currency]
        idx = bisect_left(rates, dt, key=lambda r: r[0])
        idx = min(idx, len(rates) - 1)
        return rates[idx][1]

    def convert(self, amount: "Amount", to_currency: Currency, dt: datetime) -> float:
        """Convert the monetary amount into another currency at the provided time.

        Args:
            amount (Amount): The amount to convert.
            to_currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        dt = dt.astimezone(timezone.utc)
        return amount.value * self._get_rate(to_currency, dt) / self._get_rate(amount.currency, dt)


class StaticConversion(CurrencyConverter):
    """Currency converter that uses static configured rates to convert between different currencies.
    This converter doesn't take `time` into consideration.
    """

    def __init__(self, base_currency: Currency, rates: dict[Currency, float]):
        """Initialize the StaticConversion.

        Args:
            base_currency (Currency): The base currency.
            rates (dict[Currency, float]): The static exchange rates.
        """
        super().__init__()
        self.base_currency = base_currency
        self.rates = rates
        self.rates[base_currency] = 1.0

    def convert(self, amount: "Amount", to_currency: Currency, dt: datetime) -> float:
        """Convert the monetary amount into another currency using static rates.

        Args:
            amount (Amount): The amount to convert.
            to_currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        return amount.value * self.rates[to_currency] / self.rates[amount.currency]


class One2OneConversion(CurrencyConverter):
    """Currency converter that always converts 1 to 1 between currencies, you should only use this
    for testing purposes, never real trading. So for example, 1 USD equals 1 EUR equals 1 GPB.
    """

    def convert(self, amount: "Amount", to_currency: Currency, dt: datetime) -> float:
        """Convert the monetary amount into another currency at a 1:1 rate.

        Args:
            amount (Amount): The amount to convert.
            to_currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        return amount.value


@dataclass(frozen=True, slots=True)
class Amount:
    """A monetary value denoted in a single `Currency`. Amounts are immutable."""

    currency: Currency
    value: float
    __converter: ClassVar[CurrencyConverter] = NoConversion()

    @staticmethod
    def register_converter(converter: CurrencyConverter):
        """Register a new currency converter to handle conversions between different currencies.
        It will replace the current registered converter.

        Args:
            converter (CurrencyConverter): The new currency converter.
        """
        Amount.__converter = converter

    def items(self) -> list[tuple[Currency, float]]:
        """Return a list with only this amount as the item, this brings the `Amount` class in line with the `Wallet` class.

        Returns:
            list[tuple[Currency, float]]: A list containing a tuple of the currency and value.
        """
        return [(self.currency, self.value)]

    def amounts(self) -> list["Amount"]:
        """Return a list with only this amount, this brings the `Amount` class in line with the `Wallet` class.

        Returns:
            list[Amount]: A list containing this amount.
        """
        return [self]

    def __add__(self, other: "Amount") -> "Wallet":
        """Add another amount to this amount. It will return a `Wallet`.
        So no currency conversion will be done.

        Args:
            other (Amount): The other amount to add.

        Returns:
            Wallet: The resulting Wallet object.
        """
        return Wallet(self, other)

    def __matmul__(self, other: Currency) -> "Amount":
        """Convert this amount to another currency and return a new `Amount`.

        Args:
            other (Currency): The target currency.

        Returns:
            Amount: The resulting Amount object.
        """
        dt = datetime.now(tz=timezone.utc)
        return Amount(other, self.convert_to(other, dt))

    def convert_to(self, currency: Currency, dt: datetime) -> float:
        """Convert this amount to another currency and return the monetary value.
        If an exchange rate is required, it will invoke the registered `Amount.converter` under the hood.

        Args:
            currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        if currency == self.currency:
            return self.value
        if self.value == 0.0:
            return 0.0

        return Amount.__converter.convert(self, currency, dt)

    def __repr__(self) -> str:
        """Return a string representation of the amount.

        Returns:
            str: The string representation.
        """
        return f"{self.value:,.2f}@{self.currency}"


class Wallet(defaultdict[Currency, float]):
    """A wallet holds monetary values of different currencies.
    Wallets are mutable, and you can add wallets together, subtract them, or convert them to a single currency.
    """

    def __init__(self, *amounts: Amount):
        """Initialize the Wallet with given amounts.

        Args:
            amounts (Amount): The amounts to initialize the wallet with.
        """
        super().__init__(float)
        for amount in amounts:
            self[amount.currency] += amount.value

    def amounts(self):
        """Return a list with amounts contained in this wallet.

        Returns:
            list[Amount]: A list of amounts in the wallet.
        """
        return [Amount(k, v) for k, v in self.items()]

    def __iadd__(self, other: "Amount | Wallet"):
        """Add another amount or wallet to this wallet.

        Args:
            other (Amount | Wallet): The other amount or wallet to add.

        Returns:
            Wallet: The updated wallet.
        """
        for k, v in other.items():
            self[k] += v
        return self

    def __isub__(self, other: "Amount | Wallet"):
        """Subtract another amount or wallet from this wallet.

        Args:
            other (Amount | Wallet): The other amount or wallet to subtract.

        Returns:
            Wallet: The updated wallet.
        """
        for k, v in other.items():
            self[k] -= v
        return self

    def __add__(self, other: "Amount | Wallet"):
        """Add another amount or wallet to this wallet and return a new wallet.

        Args:
            other (Amount | Wallet): The other amount or wallet to add.

        Returns:
            Wallet: The resulting wallet.
        """
        result = self.deepcopy()
        for k, v in other.items():
            result[k] += v
        return result

    def __sub__(self, other: "Amount | Wallet"):
        """Subtract another amount or wallet from this wallet and return a new wallet.

        Args:
            other (Amount | Wallet): The other amount or wallet to subtract.

        Returns:
            Wallet: The resulting wallet.
        """
        result = self.deepcopy()
        for k, v in other.items():
            result[k] -= v
        return result

    def __matmul__(self, other: Currency) -> Amount:
        """Convert all the amounts in this wallet to a single currency and return a new `Amount`.

        Args:
            other (Currency): The target currency.

        Returns:
            Amount: The resulting Amount object.
        """
        dt = datetime.now(tz=timezone.utc)
        return Amount(other, self.convert_to(other, dt))

    def deepcopy(self) -> "Wallet":
        """Return a deep copy of this wallet.

        Returns:
            Wallet: The deep copied wallet.
        """
        result = Wallet()
        result.update(self)
        return result

    def convert_to(self, currency: Currency, dt: datetime) -> float:
        """Convert all the amounts held in this wallet to a single currency and return the value.

        Args:
            currency (Currency): The target currency.
            dt (datetime): The date and time of the conversion.

        Returns:
            float: The converted monetary value.
        """
        return sum(amount.convert_to(currency, dt) for amount in self.amounts())

    def __repr__(self) -> str:
        """Return a string representation of the wallet.

        Returns:
            str: The string representation.
        """
        return " + ".join([f"{a}" for a in self.amounts()])
