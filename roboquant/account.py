from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from roboquant.order import Order
from prettytable import PrettyTable


@dataclass(slots=True, frozen=True)
class Position:
    """Position of a symbol"""

    size: Decimal
    """Position size"""

    avg_price: float
    """Average price paid denoted in the currency of the symbol"""


class Account:
    """The account maintains the following state during a run:

    - Available cash for trading (also sometimes referred to as buying power)
    - Open positions
    - Open orders
    - Total equity value of the account
    - Last time the account was updated

    Only the broker updates the state of the account and does this only during its `sync` method.
    """

    def __init__(self):
        self.buying_power: float = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.last_update: datetime = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.equity = 0.0

    def get_value(self, symbol: str, size: Decimal, price: float) -> float:
        """Return the total value of the provided contract size denoted in the base currency of the account.
        The default implementation returns `size * price`.

        The bahavior of this method can be changed for symbols denoted in a different currency and/or contract size by
        providing a different value_calculator.
        """
        return float(size) * price

    def mkt_value(self, prices: dict[str, float]) -> float:
        """Return the the market value of all the open positions in the account using the provided prices."""
        return sum([self.get_value(symbol, pos.size, prices[symbol]) for symbol, pos in self.positions.items()], 0.0)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            [self.get_value(symbol, pos.size, prices[symbol] - pos.avg_price) for symbol, pos in self.positions.items()],
            0.0,
        )

    def has_open_order(self, symbol: str) -> bool:
        """Return True if there an open order for the symbol, False otherwise"""

        for order in self.orders:
            if order.symbol == symbol and not order.closed:
                return True
        return False

    def get_position_size(self, symbol) -> Decimal:
        pos = self.positions.get(symbol)
        return pos.size if pos else Decimal(0)

    def open_orders(self):
        """Return a list with the open orders"""
        return [order for order in self.orders if not order.closed]

    def __repr__(self) -> str:
        p = PrettyTable(["account", "value"], align="r", float_format="12.2")
        p.add_row(["buying power", self.buying_power])
        p.add_row(["equity", self.equity])
        p.add_row(["positions", len(self.positions)])
        p.add_row(["orders", len(self.orders)])
        p.add_row(["last update", self.last_update.strftime("%Y-%m-%d %H:%M:%S")])
        result = p.get_string() + "\n\n"

        if self.positions:
            p = PrettyTable(["symbol", "position size", "avg price"], align="r", float_format="12.2")
            for symbol, pos in self.positions.items():
                p.add_row([symbol, pos.size, pos.avg_price])
            result += p.get_string() + "\n\n"

        if self.orders:
            p = PrettyTable(["symbol", "order size", "order id", "limit", "status", "closed"], align="r", float_format="12.2")
            for order in self.orders:
                p.add_row([order.symbol, order.size, order.id, order.limit, order.status.name, order.closed])
            result += p.get_string() + "\n"

        return result


class OptionAccount(Account):
    """
    This account handles common option contracts of size 100 and 10. Serves as an example.
    If no contract size is registered for a symbol, it creates one based on the symbol name.
    """

    def __init__(self):
        super().__init__()
        self._contract_sizes: dict[str, float] = {}

    def register(self, symbol: str, contract_size: float = 100.0):
        """Register a certain contract-size for a symbol"""
        self._contract_sizes[symbol] = contract_size

    def get_value(self, symbol: str, size: Decimal, price: float) -> float:
        contract_size = self._contract_sizes.get(symbol)

        # If nithng registered we try to defer the contract size from the symbol
        if contract_size is None:
            if len(symbol) == 21:
                # OCC compliant option symbol
                symbol = symbol[0:6].rstrip()
                contract_size = 10.0 if symbol[-1] == "7" else 100.0
            else:
                # not an option symbol
                contract_size = 1.0

        return contract_size * float(size) * price
