from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from roboquant.order import Order


@dataclass(slots=True)
class Position:
    """Position of a symbol"""

    size: Decimal
    """Position size"""

    avg_price: float
    """Average price paid denoted in the currency of the symbol"""

    mkt_price: float
    """latest market price denoted in the currency of the symbol"""


class Converter(ABC):
    """Abstraction that enables trading symbols that are denoted in different currencies and/or contact sizes"""

    @abstractmethod
    def __call__(self, symbol: str, time: datetime) -> float:
        """Return the conversion rate for the symbol at the given time"""
        ...


class OptionConverter(Converter):
    """
    This converter handles common option contracts of size 100 and 10 and serves as an example.

    If no contract size is registered for a symbol, it calculates one based on the symbol name.
    If the symbol is not recognized as an OCC compliant option symbol, it is assumed to have a
    contract size of 1.0
    """

    def __init__(self):
        super().__init__()
        self._contract_sizes: dict[str, float] = {}

    def register(self, symbol: str, contract_size: float = 100.0):
        """Register a contract-size for a symbol"""
        self._contract_sizes[symbol] = contract_size

    def __call__(self, symbol: str, time: datetime) -> float:
        contract_size = self._contract_sizes.get(symbol)

        # If no contract has been registered, we try to defer the contract size from the symbol
        if contract_size is None:
            if len(symbol) == 21:
                # OCC compliant option symbol
                symbol = symbol[0:6].rstrip()
                contract_size = 10.0 if symbol[-1] == "7" else 100.0
            else:
                # not an option symbol
                contract_size = 1.0

            self._contract_sizes[symbol] = contract_size

        return contract_size


class CurrencyConverter(Converter):
    """Supports trading in symbols that are denoted in a different currency from the base currency of the account"""

    def __init__(self, base_currency="USD", default_symbol_currency: str | None = "USD"):
        super().__init__()
        self.rates = {}
        self.base_currency = base_currency
        self.default_symbol_currency = default_symbol_currency
        self.registered_symbols: dict[str, str] = {}

    def register_symbol(self, symbol: str, currency: str):
        """Register a symbol and its denoted currency"""
        self.registered_symbols[symbol] = currency

    def register_rate(self, currency: str, rate: float):
        """Register a conversion rate from a currency to the base_currency"""
        self.rates[currency] = rate

    def __call__(self, symbol: str, _: datetime) -> float:
        currency = self.registered_symbols.get(symbol, self.default_symbol_currency)
        if not currency:
            raise ValueError(f"no currency or default_symbol_currency registered for symbol={symbol}")
        if currency == self.base_currency:
            return 1.0
        return self.rates[currency]


class Account:
    """Represents a trading account with all monetary amounts denoted in a single currency.
    The account maintains the following state during a run:

    - Available buying power for orders in the base currency of the account
    - Cash available
    - The open positions
    - Orders
    - Calculated derived equity value of the account in the base currency of the account
    - The last time the account was updated

    Only the broker updates the account and does this only during its `sync` method.
    """

    __converter: Converter | None = None

    def __init__(self):
        self.buying_power: float = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.last_update: datetime = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.cash: float = 0.0

    @staticmethod
    def register_converter(converter: Converter):
        """Register a converter"""
        Account.__converter = converter

    def contract_value(self, symbol: str, size: Decimal, price: float) -> float:
        # pylint: disable=not-callable
        """Return the total value of the provided contract size denoted in the base currency of the account."""
        rate = 1.0 if not Account.__converter else Account.__converter(symbol, self.last_update)
        return float(size) * price * rate

    def mkt_value(self) -> float:
        """Return the sum of the market values of the open positions in the account.

        The returned value is denoted in the base currency of the account.
        """
        return sum(
            [self.contract_value(symbol, pos.size, pos.mkt_price) for symbol, pos in self.positions.items()],
            0.0,
        )

    def equity(self) -> float:
        """Return the equity of the account. It calcaluates the sum of the mkt value of
        each open position and adds the available cash.

        The returned value is denoted in the base currency of the account.
        """
        return self.cash + self.mkt_value()

    def unrealized_pnl(self) -> float:
        """Return the sum of the unrealized profit and loss for the open position.

        The returned value is denoted in the base currency of the account.
        """
        return sum(
            [self.contract_value(symbol, pos.size, pos.mkt_price - pos.avg_price) for symbol, pos in self.positions.items()],
            0.0,
        )

    def has_open_order(self, symbol: str) -> bool:
        """Return True if there is at least one open order for the symbol, False otherwise"""

        for order in self.orders:
            if order.symbol == symbol and order.is_open:
                return True
        return False

    def get_open_orders(self, symbol: str) -> list[Order]:
        """Return a list of open orders for the provided symbol"""
        return [order for order in self.orders if order.is_open and order.symbol == symbol]

    def get_position_size(self, symbol: str) -> Decimal:
        """Return the position size for a symbol"""
        pos = self.positions.get(symbol)
        return pos.size if pos else Decimal(0)

    def open_orders(self):
        """Return a list with the open orders"""
        return [order for order in self.orders if order.is_open]

    def __repr__(self) -> str:
        p = [f"{v.size}@{k}" for k, v in self.positions.items()]
        p_str = ", ".join(p) or "none"

        o = [f"{o.size}@{o.symbol}" for o in self.open_orders()]
        o_str = ", ".join(o) or "none"

        result = (
            f"buying power : {self.buying_power:_.2f}\n"
            f"cash         : {self.cash:_.2f}\n"
            f"equity       : {self.equity():_.2f}\n"
            f"positions    : {p_str}\n"
            f"mkt value    : {self.mkt_value():_.2f}\n"
            f"open orders  : {o_str}\n"
            f"last update  : {self.last_update}"
        )
        return result
