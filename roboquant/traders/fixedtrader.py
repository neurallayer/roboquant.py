from decimal import Decimal
from typing import override

from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.traders.trader import Trader


class FixedTrader(Trader):
    """FixedTrader is a trader tries allocates teh available buying power
    to a long only positions for each of the provided assets.
    """

    def __init__(self, assets: list[Asset]):
        self.assets = set(assets)

    @override
    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        """Create orders for the given event, account and signals.

        Args:
            event (Event): The event to create orders for.
            account (Account): The account to create orders for.
            signals (list[Signal]): The signals to create orders for.
        """
        allocated_assets = set(account.portfolio.keys()) | {o.asset for o in account.orders}
        remaining_assets = self.assets - allocated_assets
        if not remaining_assets:
            return []

        budget = account.buying_power / len(remaining_assets)
        result = []
        for asset in remaining_assets:
            if price := event.get_price(asset):
                asset_budget = budget.convert_to(asset.currency, event.time)
                asset_cost = asset.value(Decimal(1), price)
                size = int(asset_budget / asset_cost)
                order = Order(asset, Decimal(size), price*1.1)
                result.append(order)

        return result
