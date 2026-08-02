from decimal import Decimal
from typing import override

from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.monetary import Amount
from roboquant.common.order import Order
from roboquant.common.portfolio import Position
from roboquant.common.signal import Signal
from roboquant.traders.trader import Trader


class SimpleTrader(Trader):
    """Trader only opens and close positions based on the received signals.
    It will not increase or decrease positions.
    """
    def __init__(self, max_positions: int | None = None, shorting: bool = False, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.shorting = shorting
        self.price_type = price_type
        self.n_positions = max_positions

    @override
    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        """Create orders for the given event, account and signals.

        Args:
            event (Event): The event to create orders for.
            account (Account): The account to create orders for.
            signals (list[Signal]): The signals to create orders for.
        """
        if not signals:
            return []

        orders_assets = {o.asset for o in account.orders}
        allocated_assets = set(account.portfolio.keys()) | orders_assets
        all_assets = set(event.price_items.keys())
        remaining_assets = all_assets - allocated_assets

        if remaining_assets:
            order_budget = account.buying_power / len(remaining_assets)
        else:
            order_budget = Amount(account.buying_power.currency, 0.0)

        result: dict[Asset, Order] = {}

        for signal in signals:
            asset = signal.asset

            if asset in orders_assets:
                continue

            if asset in result:
                continue

            price = event.get_price(asset, self.price_type)
            if price is None:
                continue

            pos = account.portfolio.get(asset, Position())
            if signal.is_entry_position(pos):
                if not self.shorting and signal.is_sell:
                    continue
                asset_budget = order_budget.convert_to(asset.currency, event.time)
                asset_cost = asset.value(Decimal(1), price)
                size = int((asset_budget / asset_cost) * signal.rating)
                if size:
                    result[asset] = Order(asset, Decimal(size), price)
            elif signal.is_exit_position(pos):
                result[asset] = Order(asset, -pos.size, price)

        return list(result.values())
