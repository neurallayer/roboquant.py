from roboquant.common.asset import Asset


from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(slots=True, frozen=True)
class Trade:
    """
    Represents a (partially) executed order with its filled size and execution price.
    It is immutable and can be used to track the realized PNL.

    Attributes
    ----------
        asset (Asset): The asset that was traded.
        size (Decimal): The size of the trade, positive for buy trades, negative for sell trades.
        price (float): The price at which the trade was executed, in the currency of the asset.
            So for a BUY, this is typically the asking price .
        pnl (float): The total realized profit and loss of the trade, calculated as the
        difference between the execute price and the average paid price. This include
        any fee or commission.
    """

    asset: Asset
    time: datetime
    size: Decimal
    price: float
    pnl: float
