from __future__ import annotations

from typing import override

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio


class SaxoBroker(LiveBroker):
    """Broker implementation using Saxo OpenAPI as
    described at https://www.developer.saxo/openapi/referencedocs
    """


    def __get_portfolio(self) -> Portfolio:
        """Get the open Net positions using api as described at https://www.developer.saxo/openapi/referencedocs/port/v1/netpositions"""
        ...


    def __get_orders(self) -> list[Order]:
        """Get the open orders using api as described at https://www.developer.saxo/openapi/referencedocs/port/v1/orders/get__port"""
        ...


    @override
    def _get_account(self) -> Account:
        account = Account()
        account.orders = self.__get_orders()
        account.portfolio = self.__get_portfolio()
        return account

    @override
    def _cancel_order(self, order: Order):
        """Cancel an open open order as described at https://www.developer.saxo/openapi/referencedocs/trade/v2/orders/delete__trade__orderids"""
        ...

    @override
    def _update_order(self, order: Order):
        """Modify an existing order as described at https://www.developer.saxo/openapi/referencedocs/trade/v2/orders/patch__trade"""
        ...

    @override
    def _place_order(self, order: Order):
        """place a single order, either limit or market as described at https://www.developer.saxo/openapi/referencedocs/trade/v2/orders/post__trade"""
        ...
