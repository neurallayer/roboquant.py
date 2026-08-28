from dataclasses import dataclass, asdict, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from roboquant.common.position import Position
from roboquant.common.asset import Asset
from roboquant.common.monetary import USD, Amount, Currency, Wallet
from roboquant.common.order import Order
from roboquant.common.timeframe import utcnow
from roboquant.common.trade import Trade


@dataclass(slots=True, frozen=True)
class Account:
    """Represents a trading account and is managed by the broker.
    It keeps track of the cash, positions, orders and trades.
    Only the `broker` updates the account and does this only during its `sync` method.
    """

    buying_power: Amount
    """Available buying power for orders in denoted in the base currency of the account"""

    cash: Wallet
    """
    The cash available in the account
    """

    positions: list[Position]
    """the open positions, values are denoted in the currency of the asset.
    Depending on the broker, these reflect `Netting` or `Hedging` positions.
    """

    orders: list[Order]
    """
    The open orders, each denoted in the currency of the asset.
    Each order in this list has an id assigned to it.
    """

    last_update: datetime
    """
    The last time the account was updated.
    """

    trades: list[Trade]
    """
    The trades that have been executed, each denoted in the currency of the asset.
    Not all brokers might fill these.
    """

    @property
    def base_currency(self) -> Currency:
        """Return the base currency of this account"""
        return self.buying_power.currency

    @staticmethod
    def empty(currency : Currency = USD) -> "Account":
        """Create an empty account"""
        return Account(
            buying_power=Amount(currency, 0),
            cash=Wallet(),
            positions=[],
            orders=[],
            last_update=utcnow(),
            trades=[]
        )

    @staticmethod
    def with_defaults(**kwargs: Any) -> "Account":
        """Create an account with non provided arguments having defaults"""
        return replace(Account.empty(), **kwargs)

    def convert(self, x: Wallet | Amount) -> float:
        """
        Convert a wallet or amount into the `base_currency` of the account at the `last_update` time.

        Args:
            x (Wallet | Amount): The wallet or amount to convert.

        Returns:
            float: The converted value in the base currency.
        """
        return x.convert_to(self.base_currency, self.last_update)

    def contract_value(self, asset: Asset, size: Decimal, price: float) -> float:
        """
        Contract value denoted in the base currency of the account. So if the asset is denoted in another
        currency, an automatic currency conversion will be performed.

        Args:
            asset (Asset): The asset for which to calculate the contract value.
            size (Decimal): The size of the contract.
            price (float): The price of the contract.

        Returns:
            float: The contract value in the base currency.
        """
        return asset.amount(size, price).convert_to(self.base_currency, self.last_update)

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
        Return the equity value denoted in the base currency of the account.

        Returns:
            float: The equity value in the base currency.
        """
        return self.convert(self.equity())

    def cash_value(self) -> float:
        """
        Return the cash value denoted in the base currency of the account.

        Returns:
            float: The cash value in the base currency.
        """
        return self.convert(self.cash)

    def position_amount(self, asset: Asset) -> Amount:
        """
        Return the netting position amount denoted in the currency of the asset.

        Returns:
            The position amount.
        """
        s = sum(p.mkt_value().value for p in self.positions if p.asset == asset)
        return Amount(asset.currency, s)


    def mkt_value(self) -> Wallet:
        """
        Return the sum of the market values of the open positions in the account.
        Short positions have a negative market value.

        Returns:
            Wallet: The total market value of all open positions.
        """
        result = Wallet()
        for position in self.positions:
            result += position.mkt_value()
        return result


    def realized_pnl(self, *assets: Asset) -> Wallet:
        """
        Return the sum of the realized profit and loss for trades executed in the account.
        If one or more asset is provided, limit it to those assets, otherwise include all assets.

        Returns:
            Wallet: The realized profit and loss.
        """
        result = Wallet()
        for trade in self.trades:
            if not assets or trade.asset in assets:
                result += Amount(trade.asset.currency, trade.pnl)
        return result

    def position_size(self, asset: Asset) -> Decimal:
        """
        Return the net position size for an asset, or zero if there is no open position for that asset.

        Args:
            asset (Asset): The asset for which to get the position size.

        Returns:
            Decimal: The position size as a Decimal.
        """
        result = Decimal()
        for pos in self.positions:
            if pos.asset == asset:
                result += pos.size
        return result

    def get_position(self, asset: Asset) -> Position:
        """
        Return the net position for an asset, or an empty position
        if the asset is not in the portfolio.
        The avg price and mkt price are not set.

        Args:
            asset (Asset): The asset for which to get the position size.

        Returns:
            Decimal: The position size as a Decimal.
        """
        return Position(asset, self.position_size(asset))


    def unrealized_pnl(self, *assets: Asset) -> Wallet:
        """
        Return the sum of the unrealized profit and loss for positions in the account.
        If one or more asset is provided, limit it to those assets, otherwise include all assets.

        Returns:
            Wallet: The unrealized profit and loss.
        """
        result = Wallet()
        for pos in self.positions:
            if not assets or pos.asset in assets:
                result += pos.unrealized_pnl()
        return result

    def pnl(self, *assets: Asset) -> Wallet:
        """
        Return the total profit and loss of the account, which is
        the sum of realized- and unrealized-PnL.
        If one or more asset is provided, limit it to those assets, otherwise include all assets.

        Returns:
            Wallet: The total profit and loss.
        """
        return self.realized_pnl(*assets) + self.unrealized_pnl(*assets)

    def pnl_value(self, *assets: Asset) -> float:
        """Return the total profit and loss of the account, which is
            the sum of realized- and unrealized-PnL expressed in the
            base currency of the account.
        """
        return self.convert(self.pnl())

    def get_order(self, order_id: str) -> Order | None:
        """Return an order by its id, or None if no matching order can be found"""
        for order in self.orders:
            if order.id == order_id:
                return order
        return None

    def __repr__(self) -> str:
        """Condensed representation of this account. It by default won't
        display decimals for the various amounts. But you can use the float
        formatting spec to influence this behavior: f"{account:{.4f}}"
        """
        return f"{self:,.0f}"

    def __format__(self, format_spec: str) -> str:
        """Return a float formatted string representation of the wallets
        in the account.

        Args:
            format_spec (str): The format specification.

        Returns:
            str: The formatted string representation.
        """
        p = [f"{p.size}@{p.asset.symbol}" for p in self.positions]
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

    def trades_for_asset(self, asset: Asset) -> list[Trade]:
        """Get all the trades for the provided asset"""
        return [trade for trade in self.trades if trade.asset == asset]

    def orders_for_asset(self, asset: Asset) -> list[Order]:
        """Get all the orders for the provided asset"""
        return [order for order in self.orders if order.asset == asset]

    def positions_for_asset(self, asset: Asset) -> list[Position]:
            """Get all the positions for the provided asset"""
            return [pos for pos in self.positions if pos.asset == asset]

    def positions_to_dataframe(self) -> pd.DataFrame:
        """Return the positions as a dataframe"""
        return pd.json_normalize([asdict(pos) for pos in self.positions])

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Return the trades as a dataframe"""
        return pd.json_normalize([asdict(trade) for trade in self.trades])

    def orders_to_dataframe(self) -> pd.DataFrame:
        """Return the orders as a dataframe"""
        return pd.json_normalize([asdict(order) for order in self.orders])

    def plot_allocation(self, ax: Axes | None = None, include_cash: bool = False, **kwargs: Any) -> Axes:
        """Plot the exposure of the assets in the portfolio as a pie chart.
        The allocation is based on the latest market value of the positions.

        Args:
            ax: The matplotlib axes to plot on.
            include_cash: Whether to include cash in the allocation pie chart.
            **kwargs: Additional keyword arguments to pass to the `pie()` plotting function.

        Returns:
            matplotlib.axes.Axes: The axes object with the pie chart.
        """
        if not ax:
            _, ax = plt.subplots()

        if include_cash:
            labels = ["cash"]
            sizes = [self.convert(self.cash)]
        else:
            labels = []
            sizes = []

        assets = set(p.asset for p in self.positions)
        labels = labels + [asset.symbol for asset in assets]
        sizes = sizes + [
            self.convert(abs(self.position_amount(asset))) for asset in assets
        ]

        if len(labels) == 0:
            return ax

        kwargs = {
            "autopct": "%1.1f%%",
            "labels": labels,
        } | kwargs

        ax.pie(sizes, **kwargs) # type: ignore
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        return ax
