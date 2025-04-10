from decimal import Decimal
from roboquant.account import Position
from roboquant.asset import Asset
from roboquant.monetary import USD, Amount, Wallet
from roboquant.journals.metric import Metric
from typing import Dict
from roboquant.event import Event
from roboquant.account import Account
from roboquant.signal import Signal
from roboquant.order import Order


class MarketMetric(Metric):
    """Calculates the market PNL by acquiring the same amount of all assets and sum their individual PNL performance.
    So this metrics reflects the long only performance of the market.
    """

    def __init__(self, initial_amount: Amount = USD(1_000.0), price_type: str = "DEFAULT") -> None:
        self.initial_amount = initial_amount
        self.positions: dict[Asset, Position] = {}
        self.price_type = price_type

    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> Dict[str, float]:
        for asset, item in event.price_items.items():
            price = item.price(self.price_type)
            if asset not in self.positions:
                converted_value = self.initial_amount.convert_to(asset.currency, event.time)
                size = Decimal(converted_value / price)
                self.positions[asset] = Position(size, price, price)
            else:
                self.positions[asset].mkt_price = price

        if not self.positions:
            return {
                "market/pnl" : 0.0,
                "market/avg_pnl" : 0.0,
                "market/pnl_perc" : 0.0
            }

        # Calculate the total PNL for all positions
        w = Wallet()
        for asset, position in self.positions.items():
            w += asset.contract_amount(position.size, position.mkt_price - position.avg_price)

        pnl = account.convert(w)
        avg_pnl = pnl / len(self.positions)
        pnl_perc = pnl / ( account.convert(self.initial_amount) * len(self.positions))

        return {
            "market/pnl" : pnl,
            "market/avg_pnl" : avg_pnl,
            "market/pnl_perc" : pnl_perc
        }
