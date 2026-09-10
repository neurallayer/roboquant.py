import logging
import re
from abc import ABC
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, ClassVar, override

from roboquant.common.monetary import USD, Amount, Currency

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class Asset(ABC):
    """Abstract base class for all types of assets, ranging from stocks to cryptocurrencies.
    Every asset has always at least a `symbol` and `currency` defined. Assets are immutable.

    The symbol by itself should be unique across assets and asset types. If that is
    not the case, the symbol could be extended with some additional information to make it unique.
    For example, for stocks, the exchange could be added to the symbol name.
    """

    symbol: str
    """The unique symbol name of the asset, for example, AAPL"""

    currency: Currency = USD
    """The currency of the asset, default is `USD`"""

    info: dict[str, Any] | None = None
    """Additional info that can be set"""

    _registry: ClassVar[dict[str, "Asset"]] = {}
    """Keeps track of all created assets and their symbol name"""

    def register(self, symbol: str | None = None):
        """Register the asset with the given symbol. If no symbol is provided, use the asset's symbol.
        Other components like the Feed can use the registry to look up assets by symbol.
        """
        Asset._registry[symbol or self.symbol] = self

    @classmethod
    def get_asset(cls, symbol: str, default: "Asset | None" = None) -> "Asset | None":
        """Return the asset for the given symbol. If no asset is found, return the default value."""
        return Asset._registry.get(symbol, default)

    def value(self, size: Decimal, price: float) -> float:
        """Return the total value given the provided size and price.
        The default implementation simply multiplies the size with the price, but for some asset types, like options,
        this can be overridden to include a contract size multiplier.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            float: The total contract value.
        """
        return float(size) * price

    def amount(self, size: Decimal, price: float) -> Amount:
        """Return the total amount given the provided size and price.
        The returned amount is denoted in the currency of the asset.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            Amount: The total contract amount.
        """
        if size == 0:
            return Amount(self.currency, 0.0)

        value = self.value(size, price)
        return Amount(self.currency, value)

    def __hash__(self) -> int:
        """Calculate the hash of the asset based on its symbol.

        Returns:
            The hash value of the asset.
        """
        return hash(self.symbol)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return self.symbol == value.symbol and self.currency == value.currency
        return False

    @property
    def asset_class(self) -> str:
        """Return the class of the asset, the default implementation returns the Python class name of the instance.

        Returns:
            str: The class name of the asset.
        """
        return self.__class__.__name__


@dataclass(frozen=True, slots=True)
class Stock(Asset):
    """Tradable stock or equity asset.

    A stock represents ownership in a publicly traded company. The `symbol`
    should identify the listing unambiguously within the feed or broker
    being used, for example `AAPL` or `NASDAQ:AAPL` when an exchange qualifier
    is needed. Stocks default to being denominated in `USD`, but a different
    `Currency` can be provided for non-US listings.
    """

@dataclass(frozen=True, slots=True)
class Crypto(Asset):
    """Tradable cryptocurrency asset or crypto pair.

    A crypto asset is typically expressed as a base/quote pair, such as
    `BTC/USDT` or `ETH/EUR`. The full pair is kept as the `symbol`, while the
    quote part identifies the asset currency. Use `from_symbol` for common pair
    formats so the currency can be derived from the symbol automatically.
    """

    contract_size: Decimal = Decimal(1)
    """contract or lot size"""

    @override
    def value(self, size: Decimal, price: float) -> float:
        return float(size) * price * float(self.contract_size)

    @staticmethod
    def from_symbol(symbol: str) -> "Crypto":
        """Create a Crypto asset from a symbol string. It will automatically extract
        the quote currency from the symbol, which is assumed to be the
        last part of the symbol.

        Args:
            symbol (str): The symbol string of the crypto asset.

        Returns:
            Crypto: The created crypto asset.
        """
        parts = re.split(r"[^a-zA-Z0-9\s]", symbol)
        if len(parts) == 2:
            return Crypto(symbol, Currency(parts[1]))
        if len(parts) == 1 and len(symbol) == 6:
            return Crypto(symbol, Currency(symbol[-3:]))
        raise ValueError("Cannot convert symbol %s to Forex", symbol)


@dataclass(frozen=True, slots=True)
class Forex(Asset):
    """Foreign exchange currency-pair asset.

    A forex asset represents one currency traded against another, usually
    written as a base/quote pair such as `EUR/USD` or `GBP/JPY`. The complete
    pair is stored as the `symbol`, and the quote currency is used as the
    asset currency. Use `from_symbol` for standard pair notation so the quote
    currency can be inferred automatically.
    """

    contract_size: Decimal = Decimal(1)
    """contract or lot size"""

    @override
    def value(self, size: Decimal, price: float) -> float:
        return float(size) * price * float(self.contract_size)

    @staticmethod
    def from_symbol(symbol: str) -> "Forex":
        """Create a `Forex` asset from a symbol string.
        The last part of the symbol name is assumed to be the quote currency.
        Any non-alphanumeric character is considered to be a separator.

        Args:
            symbol (str): The symbol string of the forex asset.
        Returns:
            Forex: The created asset of the type Forex.
        """
        parts = re.split(r"[^a-zA-Z0-9\s]", symbol)
        if len(parts) == 2:
            return Forex(symbol, Currency(parts[1]))
        if len(parts) == 1 and len(symbol) == 6:
            return Forex(symbol, Currency(symbol[-3:]))
        raise ValueError("Cannot convert symbol %s to Forex", symbol)

@dataclass(frozen=True, slots=True)
class Option(Asset):
    """Option contract asset with a standard contract multiplier.

    An option represents the right to buy or sell an underlying asset at a
    specified strike and expiry. The `symbol` should identify the contract
    unambiguously, for example using an OCC-style option symbol.

    Option prices are quoted per underlying unit, so this class multiplies contract value by
    `100` to model standard equity option contract sizes. This multiplier cannot be changed.
    """

    @override
    def value(self, size: Decimal, price: float) -> float:
        """Contract value for this option type is the `size` times the `price` times `100`.

        Args:
            size (Decimal): The size of the contract.
            price (float): The price per unit of the asset.

        Returns:
            float: The total contract value.
        """
        return float(size) * price * 100.0

    def decode_occ_symbol(self) -> dict[str, Any]:
        """
        Decode the symbol into its components.

        Standard format:
            ROOT + YYMMDD + C/P + STRIKE
        where STRIKE is usually 8 digits (strike * 1000, padded with zeros),
        but a 5‑digit variant (strike * 100) is also supported.

        Returns:
            dict: {
                'underlying': str,
                'expiration': datetime.date,
                'option_type': str,   # 'C' or 'P'
                'strike': float
            }

        Raises:
            ValueError: If the symbol does not match the expected format.
        """
        # Strip any leading/trailing whitespace
        symbol = self.symbol.strip()

        # Regular expression to capture the main parts.
        # Root: one or more uppercase letters
        # Expiry: 6 digits (YYMMDD)
        # Type: C or P
        # Strike: 5 or 8 digits (we'll treat both)
        pattern = r"^([A-Z]+)(\d{6})([CP])(\d{5,8})$"
        match = re.match(pattern, symbol)
        if not match:
            raise ValueError(f"Invalid OCC symbol format: {symbol}")

        root, expiry_str, option_type, strike_str = match.groups()

        # Parse expiration date
        try:
            decade = int(expiry_str[:2])
            year = 2000 + decade if decade < 70 else 1900 + decade
            month = int(expiry_str[2:4])
            day = int(expiry_str[4:6])
            expiration = date(year, month, day)
        except ValueError as e:
            raise ValueError(f"Invalid expiration date in symbol: {expiry_str}") from e

        # Decode strike price
        # If length is 8, divide by 1000; if length is 5, divide by 100.
        # (Some older or non‑OCC sources may use 5 digits.)
        if len(strike_str) == 8:
            strike = int(strike_str) / 1000.0
        elif len(strike_str) == 5:
            strike = int(strike_str) / 100.0
        else:
            # This case shouldn't happen due to regex, but just in case
            raise ValueError(f"Unexpected strike length: {len(strike_str)}")

        return {"underlying": root, "expiration": expiration, "option_type": option_type, "strike": strike}
