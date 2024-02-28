import inspect
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from roboquant.order import Order


@dataclass(slots=True, frozen=True)
class Position:
    """Position of a symbol"""

    size: Decimal
    """Position size"""

    avg_price: float
    """Average price paid denoted in the currency of the symbol"""


class Account:
    """The account maintains the following state during a run:

    - Available buying power for orders in the base currency of the account
    - All the open positions
    - Orders
    - Total equity value of the account in the base currency of the account
    - The last time the account was updated

    Only the broker updates the account and does this only during its `sync` method.
    """

    buying_power: float
    positions: dict[str, Position]
    orders: list[Order]
    last_update: datetime
    equity: float

    def __init__(self):
        self.buying_power: float = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.last_update: datetime = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.equity: float = 0.0

    def contract_value(self, symbol: str, size: Decimal, price: float) -> float:
        """Return the total value of the provided contract size denoted in the base currency of the account.
        The default implementation returns `size * price`.

        A subclass can implement different logic to cater for:
        - symbols denoted in different currencies
        - symbols having different contract sizes like option contracts.
        """
        return float(size) * price

    def mkt_value(self, prices: dict[str, float]) -> float:
        """Return the market value of all the open positions in the account using the provided prices.
        If there is no known price provided for a position, the average price paid will be used instead.

        Args:
            prices: The prices to use to calculate the market value.
        """
        return sum(
            [
                self.contract_value(symbol, pos.size, prices[symbol] if symbol in prices else pos.avg_price)
                for symbol, pos in self.positions.items()
            ],
            0.0,
        )

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Return the unrealized profit and loss for the open position given the provided market prices
        If there is no known price provided for a position, it will be ignored.

        Args:
            prices: The prices to use to calculate the unrealized PNL.
        """
        return sum(
            [
                self.contract_value(symbol, pos.size, prices[symbol] - pos.avg_price)
                for symbol, pos in self.positions.items()
                if symbol in prices
            ],
            0.0,
        )

    def has_open_order(self, symbol: str) -> bool:
        """Return True if there is an open order for the symbol, False otherwise"""

        for order in self.orders:
            if order.symbol == symbol and not order.closed:
                return True
        return False

    def get_position_size(self, symbol) -> Decimal:
        """Return the position size for the symbol"""
        pos = self.positions.get(symbol)
        return pos.size if pos else Decimal(0)

    def open_orders(self):
        """Return a list with the open orders"""
        return [order for order in self.orders if not order.closed]

    def __repr__(self) -> str:
        p = [f"{v.size}@{k}" for k, v in self.positions.items()]
        p_str = ", ".join(p) or "none"

        o = [f"{o.size}@{o.symbol}" for o in self.open_orders()]
        o_str = ", ".join(o) or "none"

        result = f"""
            buying power : {self.buying_power:_.2f}
            equity       : {self.equity:_.2f}
            positions    : {p_str}
            open orders  : {o_str}
            last update  : {self.last_update}
        """
        return inspect.cleandoc(result)


class OptionAccount(Account):
    """
    This account handles common option contracts of size 100 and 10 and serves as an example.
    If no contract size is registered for a symbol, it creates one based on the option symbol name.

    If the symbol is not recognized as an OCC compliant option symbol, it is assumed to have a
    contract size of 1.0
    """

    def __init__(self):
        super().__init__()
        self._contract_sizes: dict[str, float] = {}

    def register(self, symbol: str, contract_size: float = 100.0):
        """Register a certain contract-size for a symbol"""
        self._contract_sizes[symbol] = contract_size

    def contract_value(self, symbol: str, size: Decimal, price: float) -> float:
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

        return contract_size * float(size) * price
