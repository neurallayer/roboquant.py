from decimal import Decimal
import logging
from typing import override

from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.monetary import Amount
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.traders.trader import Trader


logger = logging.getLogger(__name__)


class SimpleTrader(Trader):
    """Trader only opens and close positions based on the received signals using Market Orders.
    It will not increase or decrease position sizes once a position is opened for an asset.
    If there is already an open order for an asset, it will not create another one.
    But the open orders don't count for positions.
    """
    def __init__(self, max_positions: int = 10, shorting: bool = False, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self.shorting = shorting
        self.price_type = price_type
        self.max_positions: int = max_positions

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

        remaining_positions = self.max_positions - len(account.positions)
        order_assets = {o.asset for o in account.orders}

        if remaining_positions > 0:
            order_budget = account.buying_power / remaining_positions
        else:
            order_budget = Amount(account.buying_power.currency, 0.0)

        result: dict[Asset, Order] = {}

        for signal in signals:
            asset = signal.asset

            if asset in order_assets:
                logger.info("existing order exists")
                continue

            if asset in result:
                logger.info("multiple signals for same asset")
                continue

            price = event.get_price(asset, self.price_type)
            if price is None:
                logger.info("no price found")
                continue

            pos = account.get_position(asset)
            if signal.is_open_position(pos):

                if remaining_positions <= 0:
                    logger.info("no remaining positions")
                    continue

                if not self.shorting and signal.is_sell:
                    logger.info("shorting not allowed")
                    continue

                asset_budget = order_budget.convert_to(asset.currency, event.time)
                asset_cost = asset.value(Decimal(1), price)
                size = int((asset_budget / asset_cost) * signal.rating)
                if size:
                    result[asset] = Order(asset, Decimal(size))
                    remaining_positions -= 1
            elif signal.is_close_position(pos):
                result[asset] = pos.close_order()

        return list(result.values())
