from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from roboquant.asset import Asset
from roboquant.monetary import USD, Amount, Currency, Wallet
from roboquant.order import Order


@dataclass(slots=True, frozen=True)
class Trade:
    """
    A trade represents a (partially) executed order with its filled size and execution price.
    It is immutable and can be used to track the execution of an order.

    Attributes:
        asset (Asset): The asset that was traded.
        size (Decimal): The size of the trade, positive for buy trades, negative for sell trades.
        price (float): The price at which the trade was executed, in the currency of the asset.
        pnl (float): The profit and loss of the trade, calculated as the difference between the trade price and the average
        paid price.
    """

    asset: Asset
    time: datetime
    size: Decimal
    price: float
    pnl: float


@dataclass(slots=True)
class Position:
    """The position of an asset in the account. The position prices are denoted in the currency of the asset that
    is linked to this posistion. See also :meth:`roboquant.Account.positions`.
    """

    size: Decimal
    """Position size as a Decimal with a negative size indicating a short position"""

    avg_price: float
    """Average price paid denoted in the currency of the asset"""

    mkt_price: float
    """Latest market price denoted in the currency of the asset"""

    @property
    def is_short(self):
        """Return True if this is a short position, False otherwise"""
        return self.size < 0

    @property
    def is_long(self):
        """Return True if this is a long position, False otherwise"""
        return self.size > 0

    @staticmethod
    def zero():
        """Return a zero position size with no known prices"""
        return Position(Decimal(), float("nan"), float("nan"))

@dataclass
class Account:
    """Represents a trading account. The account maintains the following state during a run:

    Attributes:
        buying_power (Amount): Available buying power for orders in the base currency of the account.
        cash (Wallet): The cash available in the account.
        positions (Dict[Asset, Position]): the open positions, each denoted in the currency of the asset.
        orders (list[Order]): the open orders, each denoted in the currency of the asset.
        trades (list[Trade]): the trades that have been executed, each denoted in the currency of the asset.
        last_update (datetime): The last time the account was updated.

    Only the `broker` updates the account and does this only during its `sync` method.
    """

    __slots__ = "buying_power", "positions", "orders", "last_update", "cash", "trades"

    def __init__(self, base_currency: Currency = USD):
        """
        Initialize a new Account instance.

        Args:
            base_currency (Currency): The base currency of the account, defaults to USD.
        """
        self.buying_power: Amount = Amount(base_currency, 0.0)
        self.positions: dict[Asset, Position] = {}
        self.orders: list[Order] = []
        self.last_update: datetime = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.cash: Wallet = Wallet()
        self.trades: list[Trade] = []

    @property
    def base_currency(self) -> Currency:
        """Return the base currency of this account"""
        return self.buying_power.currency

    def mkt_value(self) -> Wallet:
        """
        Return the sum of the market values of the open positions in the account. Short
        positions have a negative market value.

        Returns:
            Wallet: The total market value of all open positions.
        """
        result = Wallet()
        for asset, position in self.positions.items():
            result += asset.contract_amount(position.size, position.mkt_price)
        return result

    def convert(self, x: Wallet | Amount) -> float:
        """
        Convert a wallet or amount into the `base_currency` of the account at the `last_update` time.

        Args:
            x (Wallet | Amount): The wallet or amount to convert.

        Returns:
            float: The converted value in the base currency.
        """
        return x.convert_to(self.base_currency, self.last_update)

    def position_value(self, asset: Asset) -> float:
        """
        Return position value denoted in the base currency of the account. If there is no
        open position, 0.0 wil lbe returned. Short positions will return a negative value.

        Args:
            asset (Asset): The asset for which to calculate the position value.

        Returns:
            float: The position value in the base currency.
        """
        pos = self.positions.get(asset)
        return asset.contract_value(pos.size, pos.mkt_price) if pos else 0.0

    def short_positions(self) -> dict[Asset, Position]:
        """
        Return all the open short positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding short positions.
        """
        return {asset: position for (asset, position) in self.positions.items() if position.is_short}

    def long_positions(self) -> dict[Asset, Position]:
        """
        Return all the open long positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding long positions.
        """
        return {asset: position for (asset, position) in self.positions.items() if position.is_long}

    def contract_value(self, asset: Asset, size: Decimal, price: float) -> float:
        """
        Contract value denoted in the base currency of the account. So if the asset is denoted in another
        currency, an automatic currency conversion will done.

        Args:
            asset (Asset): The asset for which to calculate the contract value.
            size (Decimal): The size of the contract.
            price (float): The price of the contract.

        Returns:
            float: The contract value in the base currency.
        """
        return asset.contract_amount(size, price).convert_to(self.base_currency, self.last_update)

    def equity(self) -> Wallet:
        """
        Return the equity of the account.
        It calculates the sum of market values of each open position and adds the available cash.

        Returns:
            Wallet: The equity of the account.
        """
        return self.cash + self.mkt_value()

    def equity_value(self) -> float:
        """
        Return the equity value denoted in the base currency of the account.x

        Returns:
            float: The equity value in the base currency.
        """
        return self.convert(self.equity())

    def unrealized_pnl(self) -> Wallet:
        """
        Return the sum of the unrealized profit and loss for the open positions.

        Returns:
            Wallet: The unrealized profit and loss.
        """
        result = Wallet()
        for asset, position in self.positions.items():
            result += asset.contract_amount(position.size, position.mkt_price - position.avg_price)
        return result

    def realized_pnl(self) -> Wallet:
        """
        Return the sum of the realized profit and loss for trades executed in the account.

        Returns:
            Wallet: The realized profit and loss.
        """
        result = Wallet()
        for trade in self.trades:
            result += Amount(trade.asset.currency, trade.pnl)
        return result

    def pnl(self) -> Wallet:
        """
        Return the total profit and loss of the account, which is
        the sum of realized and unrealized PnL.

        Returns:
            Wallet: The total profit and loss.
        """
        return self.realized_pnl() + self.unrealized_pnl()

    def unrealized_pnl_value(self) -> float:
        """
        Return the unrealized profit and loss value denoted in the base currency of the account.

        Returns:
            float: The unrealized profit and loss value in the base currency.
        """
        return self.convert(self.unrealized_pnl())

    def get_position_size(self, asset: Asset) -> Decimal:
        """
        Return the position size for an asset, or zero if there is no open position for that asset.

        Args:
            asset (Asset): The asset for which to get the position size.

        Returns:
            Decimal: The position size as a Decimal.
        """
        pos = self.positions.get(asset)
        return pos.size if pos else Decimal()

    def get_order(self, order_id: str) -> Order | None:
        """Return an order by its id, or None if no matching order can be found"""
        for order in self.orders:
            if order.id == order_id:
                return order

    def __repr__(self) -> str:
        """Condensed representation of this account. It by default won't
        display decimals for the various amounts. But you can use the float
        formatting spec to influence this behavior: f"{account:{.4f}}"
        """
        return f"{self:,.0f}"

    def __format__(self, format_spec: str) -> str:
        """Return a float formatted string representation of the wallet.

        Args:
            format_spec (str): The format specification.

        Returns:
            str: The formatted string representation.
        """
        p = [f"{v.size}@{k.symbol}" for k, v in self.positions.items()]
        p_str = ", ".join(p) or "none"

        o = [f"{o.size}@{o.asset.symbol}" for o in self.orders]
        o_str = ", ".join(o) or "none"

        mkt = self.mkt_value() or Amount(self.base_currency, 0.0)

        result = (
            f"buying power : {self.buying_power:{format_spec}}\n"
            f"cash         : {self.cash:{format_spec}}\n"
            f"equity       : {self.equity():{format_spec}}\n"
            f"positions    : {p_str}\n"
            f"trades       : {len(self.trades)}\n"
            f"mkt value    : {mkt:{format_spec}}\n"
            f"orders       : {o_str}\n"
            f"last update  : {self.last_update}"
        )
        return result
