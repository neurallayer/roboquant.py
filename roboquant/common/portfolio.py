from roboquant.common.asset import Asset
from roboquant.common.monetary import Amount, Wallet

import pandas as pd

from collections import UserDict
from dataclasses import asdict, dataclass
from decimal import Decimal

from roboquant.common.order import Order


@dataclass(slots=True, frozen=True)
class Position:
    """The position of an asset in the portfolio. The position prices are denoted in the currency of the asset that
    is linked to this position.

    A position is immutable and is managed only by the broker.
    """

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


class Portfolio(UserDict[Asset, Position]):
    """Contains all the open postions with the corresponding asset"""

    def value(self, asset: Asset) -> Amount:
        """
        Return position amount denoted in the base currency of the account. If there is no
        open position, 0.0 will be returned. Short positions will return a negative value.

        Args:
            asset (Asset): The asset for which to calculate the position value.

        Returns:
            float: The position value in the base currency.
        """
        pos = self.get(asset, Position())
        return asset.amount(pos.size, pos.mkt_price)

    def short_positions(self) -> "Portfolio":
        """
        Return all the open short positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding short positions.
        """
        return Portfolio({asset: position for (asset, position) in self.items() if position.is_short})

    def long_positions(self) -> "Portfolio":
        """
        Return all the open long positions in the account.

        Returns:
            dict[Asset, Position]: A dictionary of assets and their corresponding long positions.
        """
        return Portfolio({asset: position for (asset, position) in self.items() if position.is_long})

    def unrealized_pnl(self) -> Wallet:
        """
        Return the sum of the unrealized profit and loss for the open positions.
        This includes both long- and short-positions.

        Returns:
            Wallet: The unrealized profit and loss.
        """
        result = Wallet()
        for asset, position in self.items():
            result += asset.amount(position.size, position.mkt_price - position.avg_price)
        return result

    def mkt_value(self) -> Wallet:
        """
        Return the sum of the market values of the open positions in the account. Short
        positions have a negative market value.

        Returns:
            Wallet: The total market value of all open positions.
        """
        result = Wallet()
        for asset, position in self.items():
            result += asset.amount(position.size, position.mkt_price)
        return result

    def close_positions(self, ndigits: int = 2) -> list[Order]:
        """Create the orders required to close the current open positions.
        The limit price of the orders is equal to the last known market price.

        Attributes:
           ndigits: how many digits to use for the limit price
        """
        orders = [Order(asset, -pos.size, round(pos.mkt_price, ndigits)) for asset, pos in self.items()]
        return orders

    def to_dataframe(self) -> pd.DataFrame:
        """Return the positions as a dataframe"""
        return pd.json_normalize([asdict(asset) | asdict(pos) for asset, pos in self.items()])

    def size(self, asset: Asset) -> Decimal:
        """
        Return the position size for an asset, or zero if there is no open position for that asset.

        Args:
            asset (Asset): The asset for which to get the position size.

        Returns:
            Decimal: The position size as a Decimal.
        """
        pos = self.get(asset)
        return pos.size if pos else Decimal()
