from roboquant.journals.metric import Metric
from typing import Dict
from roboquant.event import Event
from roboquant.account import Account
from roboquant.signal import Signal
from roboquant.order import Order


class PriceItemMetric(Metric):
    """Tracks the price and volume of individual assets found in the event."""

    def __init__(self, *symbols: str, price_type="DEFAULT", volume_type="DEFAULT"):
        """Initialize PriceMetric with specified symbols and price/volume types.
        Args:
            *symbols: Variable length argument list of str symbols to track. If none are provided,
            all encountered symbols will be included.
            price_type: Type of price to use for calculations. Defaults to "DEFAULT".
            volume_type: Type of volume to use for calculations. Defaults to "DEFAULT".
        Returns:
            None
        Examples:
            >>> metric = PriceMetric("AAPL", "MSFT", price_type="CLOSE")
        """

        super().__init__()
        self.symbols = symbols
        self.price_type = price_type
        self.volume_type = volume_type

    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> Dict[str, float]:
        result = {}
        for asset, item in event.price_items.items():
            symbol = asset.symbol
            if symbol in self.symbols or not self.symbols:
                prefix = f"item/{symbol.lower()}"
                result[f"{prefix}/price"] = item.price(self.price_type)
                result[f"{prefix}/volume"] = item.volume(self.volume_type)

        return result
