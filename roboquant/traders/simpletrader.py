from decimal import Decimal
from typing import override

from roboquant.common.account import Account
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
    def __init__(self, n_positions: int | None = None, shorting: bool = False, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.shorting = shorting
        self.price_type = price_type
        self.n_positions = n_positions

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

        allocated_assets = set(account.portfolio.keys()) | {o.asset for o in account.orders}
        all_assets = set(event.price_items.keys())
        remaining_assets = all_assets - allocated_assets

        if remaining_assets:
            order_budget = account.buying_power / len(remaining_assets)
        else:
            order_budget = Amount(account.buying_power.currency, 0.0)

        result = []

        for signal in signals:
            asset = signal.asset
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
                    order = Order(asset, Decimal(size), price)
                    result.append(order)
            elif signal.is_exit_position(pos):
                order = Order(asset, -pos.size, price)
                result.append(order)

        return result
