from decimal import Decimal
from roboquant.account import Position
from roboquant.asset import Asset
from roboquant.monetary import USD, Amount, Wallet
from roboquant.journals.metric import Metric


class MarketMetric(Metric):
    """Calculates the market PNL by acquiring the same amount of all assets and sum their individual PNL performance.
    So this is a long only performance of the market.
    """

    def __init__(self, initial_amount: Amount = USD(1_000.0), price_type: str = "DEFAULT") -> None:
        self.initial_amount = initial_amount
        self.positions: dict[Asset, Position] = {}
        self.price_type = price_type

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        for asset, item in event.price_items.items():
            price = item.price(self.price_type)
            if asset not in self.positions:
                converted_value = self.initial_amount.convert_to(asset.currency, event.time)
                size = Decimal(converted_value/price)
                self.positions[asset] = Position(size, price, price)
            else:
                self.positions[asset].mkt_price = price

        # Calculate the total PNL for all positions
        w = Wallet()
        for asset, position in self.positions.items():
            w += asset.contract_amount(position.size, position.mkt_price - position.avg_price)

        return {
            "market/pnl" : account.convert(w)
        }
