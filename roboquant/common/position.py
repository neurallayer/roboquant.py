from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from roboquant.common.asset import Asset
from roboquant.common.monetary import Amount
from roboquant.common.order import Order


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

    id: str | None = None
    """Identifier for the position, can be used for hedging accounts thatcan have
    multiple positions for a single asset."""

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

    def close_order(self, limit : float|None = None, tif: Literal['GTC', 'DAY'] = "DAY") -> Order:
        """Create a close order for this position, optionally provide a limit and Time In Force."""
        return Order(self.asset, - self.size, limit=limit, tif=tif, position_id=self.id)
