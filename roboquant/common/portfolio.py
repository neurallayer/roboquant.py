from dataclasses import asdict, dataclass
from decimal import Decimal

from roboquant.common.asset import Asset
from roboquant.common.monetary import Amount, Wallet
from roboquant.common.order import Order

import pandas as pd


@dataclass(slots=True, frozen=True)
class Position:
    """The position of an asset in the portfolio. The position prices are denoted in the
    currency of the asset that is linked to this position.

    A position object is immutable and is managed only by the broker.
    """
    asset: Asset
    """The asset of the position"""

    size: Decimal = Decimal()
    """Position size as a Decimal with a negative size indicating a short position"""

    avg_price: float = 0.0
    """Average price paid denoted in the currency of the asset"""

    mkt_price: float = float("nan")
    """Latest market price denoted in the currency of the asset. This is updated at every step in a run."""

    @property
    def is_short(self):
        """Return True if this is a short position, False otherwise"""
        return self.size < 0

    @property
    def is_long(self):
        """Return True if this is a long position, False otherwise"""
        return self.size > 0

    @property
    def is_closed(self):
        """Return True if this is a closed position, False otherwise"""
        return self.size.is_zero()

    def mkt_value(self) -> Amount:
        """
        Return the market value of the open position.
        Short positions have a negative market value.

        Returns:
            The total market value of all open positions.
        """
        return self.asset.amount(self.size, self.mkt_price)

    def unrealized_pnl(self) -> Amount:
        """
        Return the unrealized profit and loss for the open position.
        Returns:
            The unrealized profit and loss.
        """
        return self.asset.amount(self.size, self.mkt_price - self.avg_price)


class Portfolio(list[Position]):
    """Contains the open postions with the corresponding asset.
    If a position is closed, it will no longer be included in the portfolio.
    """

    def value(self, asset: Asset) -> Amount:
        """
        Return position amount denoted in the currency of the asset. If there is no
        open position, 0.0 will be returned. Short positions will return a negative value.

        Args:
            asset (Asset): The asset for which to calculate the position value.

        Returns:
            float: The position value in the base currency.
        """
        pos = self.get_position(asset)
        return asset.amount(pos.size, pos.mkt_price)

    def short_positions(self) -> "Portfolio":
        """
        Return all the open short positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding short positions.
        """
        return Portfolio(position for position in self if position.is_short)

    def long_positions(self) -> "Portfolio":
        """
        Return all the open long positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding long positions.
        """
        return Portfolio(position for position in self if position.is_long)

    def unrealized_pnl(self, *assets: Asset) -> Wallet:
        """
        Return the sum of the unrealized profit and loss for the open positions.
        This includes both long- and short-positions.
        If one or more asset is provided, limit it to those assets, otherwise include all assets.

        Returns:
            Wallet: The unrealized profit and loss.
        """
        result = Wallet()
        for position in self:
            if not assets or position.asset in assets:
                result += position.unrealized_pnl()
        return result

    def mkt_value(self, *assets: Asset) -> Wallet:
        """
        Return the sum of the market values of the open positions in the account.
        Short positions have a negative market value.
        If one or more asset is provided, limit it to those assets, otherwise
        include all assets.

        Returns:
            Wallet: The total market value of all open positions.
        """
        result = Wallet()
        for position in self:
            if not assets or position.asset in assets:
                result += position.mkt_value()
        return result


    def close_positions(self) -> list[Order]:
        """Create the market orders required to close the current open positions.
        """
        orders = [Order(pos.asset, -pos.size) for pos in self]
        return orders

    def to_dataframe(self) -> pd.DataFrame:
        """Return the positions as a dataframe"""
        return pd.json_normalize([asdict(pos) for pos in self])

    def size(self, asset: Asset) -> Decimal:
        """
        Return the net position size for an asset, or zero if there is no open position for that asset.

        Args:
            asset (Asset): The asset for which to get the position size.

        Returns:
            Decimal: The position size as a Decimal.
        """
        result = Decimal()
        for pos in self:
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
        return Position(asset, self.size(asset))
