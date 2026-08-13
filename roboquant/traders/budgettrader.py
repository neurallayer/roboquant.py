from decimal import Decimal
import logging
from typing import override

from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.monetary import Amount
from roboquant.common.order import Order
from roboquant.common.portfolio import Position
from roboquant.common.signal import Signal
from roboquant.traders.trader import Trader
from roboquant.traders._util import round_down


logger = logging.getLogger(__name__)


class BudgetTrader(Trader):
    """Trader will create orders based on a set amount per order and per position.
    All budgets are expressed in absolute values and are expected in the same
    currency as the account.
    """

    def __init__(
        self,
        order_value: float,
        position_value: float,
        shorting: bool = False,
        price_type: str = "DEFAULT",
        ndigits: int = 0
    ) -> None:
        super().__init__()
        self.shorting = shorting
        self.price_type = price_type
        self.order_value: float = order_value
        self.position_value: float = position_value
        self.ndigits = ndigits

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

        buying_power: float = account.buying_power.value
        base_currency = account.base_currency
        order_budget: Amount = Amount(base_currency, self.order_value)
        order_assets = {o.asset for o in account.orders}

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

            pos = account.portfolio.get(asset, Position())
            if signal.is_increase_position(pos):
                if buying_power < self.order_value:
                    logger.info("not enough buying_power remaining")
                    continue

                if pos.size.is_zero() and signal.is_sell and not self.shorting:
                    logger.info("shorting not allowed")
                    continue

                pos_value = account.contract_value(asset, pos.size, price)
                if pos_value >= self.position_value:
                    logger.info("max position value reached")
                    continue

                asset_budget = order_budget.convert_to(asset.currency, event.time)
                asset_cost = asset.value(Decimal(1), price)
                order_size = round_down((asset_budget / asset_cost) * signal.rating, self.ndigits)
                if order_size:
                    order = Order(asset, order_size, price)
                    result[asset] = order
                    value = order.amount().convert_to(base_currency, event.time)
                    buying_power -= abs(value)
            else:
                assert pos.size, "position size should be non-zero"
                result[asset] = Order(asset, -pos.size, price)

        return list(result.values())
