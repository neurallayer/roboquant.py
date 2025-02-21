from roboquant.journals.metric import Metric


class PriceItemMetric(Metric):
    """Tracks the price and volume of individual price-items found in the event.
    """

    def __init__(self, *symbols: str, price_type="DEFAULT", volume_type="DEFAULT"):
        super().__init__()
        self.symbols = symbols
        self.price_type = price_type
        self.volume_type = volume_type

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        result = {}
        for asset, item in event.price_items.items():
            symbol = asset.symbol
            if symbol in self.symbols or not self.symbols:
                prefix = f"item/{symbol.lower()}"
                result[f"{prefix}/price"] = item.price(self.price_type)
                result[f"{prefix}/volume"] = item.volume(self.volume_type)

        return result
