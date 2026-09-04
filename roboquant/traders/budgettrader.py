from roboquant.traders._util import Sizing, round_number
from decimal import Decimal
import logging
from typing import override

from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.monetary import Amount
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.traders.trader import Trader


logger = logging.getLogger(__name__)


class BudgetTrader(Trader):
    """
    Trader will create orders based on a set amount per order and per position.
    All budgets are expressed in absolute float values and are expected in the same
    currency as the account.
    """

    def __init__(
        self,
        order_value: float,
        position_value: float,
        shorting: bool = False,
        price_type: str = "DEFAULT",
        step_size: str = "1"
    ) -> None:
        super().__init__()
        self.shorting = shorting
        self.price_type = price_type
        self.order_value: float = order_value
        self.position_value: float = position_value
        self.step_size = step_size

    @override
    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        """
        Create orders for the given event, account and signals.

        Args:
            event (Event): The event to create orders for.
            account (Account): The account to create orders for.
            signals (list[Signal]): The signals to create orders for.
        """
        if not signals:
            return []

        buying_power: float = account.buying_power.value
        base_currency = account.base_currency
        order_budget: Amount = Amount(base_currency, self.order_value)
        order_assets = {o.asset for o in account.orders}

        result: list[Order] = []

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

            pos_size = account.position_size(asset)
            si = Sizing(signal, pos_size)

            if si.is_increase():
                if buying_power < self.order_value:
                    logger.info("not enough buying_power remaining")
                    continue

                if si.is_shorting() and not self.shorting:
                    logger.info("shorting not allowed")
                    continue

                pos_value = account.contract_value(asset, pos_size, price)
                if pos_value >= self.position_value:
                    logger.info("max position value reached")
                    continue

                asset_budget = order_budget.convert_to(asset.currency, event.time)
                asset_cost = asset.value(Decimal(1), price)
                order_size = round_number((asset_budget / asset_cost) * signal.rating, self.step_size)
                if order_size:
                    order = Order(asset, order_size)
                    result.append(order)
                    value = order.remaining_amount(price).convert_to(base_currency, event.time)
                    buying_power -= abs(value)
            else:
                assert pos_size, "position size should be non-zero"
                for pos in si.close_positions(account.positions):
                    result.append(pos.close_order())

        return result
