from __future__ import annotations

from decimal import Decimal
from typing import Any

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.monetary import USD, Amount, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position

from saxo_api_client.contrib.client import SaxoClient


class SaxoBroker(LiveBroker):
    """Live or simulation broker backed by saxo-api-client."""

    def __init__(
        self,
        access_token: str,
        account_key: str | None = None,
        client: SaxoClient | None = None,
    ) -> None:
        super().__init__()
        self.__client = client or SaxoClient(access_token=access_token)
        self.__account_key = account_key

    @staticmethod
    def __asset(row: dict[str, Any]) -> Asset:
        symbol = (
            row.get("symbol")
            or row.get("Symbol")
            or row.get("description")
            or row.get("Description")
        )
        if not symbol:
            uic = row.get("uic") or row.get("Uic")
            symbol = f"uic:{uic}"

        return Asset(symbol)

    @staticmethod
    def __value(data: dict[str, Any], *names: str) -> Any:
        for name in names:
            if data.get(name) is not None:
                return data[name]
        return None

    def __order_kwargs(self, order: Order) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "asset_type": getattr(order.asset, "asset_type", "Stock"),
            "amount": abs(float(order.size)),
            "buy_sell": "Buy" if order.is_buy else "Sell",
            "is_force_open": True,
        }

        uic = getattr(order.asset, "uic", None)
        if uic is not None:
            kwargs["uic"] = uic
        else:
            kwargs["symbol"] = order.asset.symbol

        return kwargs

    def _sync_positions(self) -> Portfolio:
        portfolio = Portfolio()

        for row in self.__client.get_positions():
            quantity = Decimal(str(row.get("quantity", 0)))
            if not quantity:
                continue

            portfolio[self.__asset(row)] = Position(
                quantity,
                float(row.get("open_price", 0)),
                float(row.get("current_price", "nan")),
            )

        return portfolio

    def _sync_orders(self) -> list[Order]:
        orders: list[Order] = []
        return orders

        for row in self.__client.get_active_orders():
            amount = Decimal(str(row.get("amount", row.get("Amount", 0))))
            filled = Decimal(str(row.get("filled_amount", row.get("FilledAmount", 0))))
            limit = row.get("order_price", row.get("OrderPrice"))
            tif = row.get("time_in_force", "DAY")
            order_id = str(row.get("order_id", row.get("OrderId")))

            if row.get("buy_sell", row.get("BuySell", "Buy")) == "Buy":
                orders.append(
                    self._buy_order(
                        order_id, self.__asset(row), amount, limit, filled, tif
                    )
                )
            else:
                orders.append(
                    self._sell_order(
                        order_id, self.__asset(row), amount, limit, filled, tif
                    )
                )

        return orders

    def _get_account(self) -> Account:
        account = Account()
        summary = self.__client.summarize_client_netting()

        cash = self.__value(summary, "cash", "Cash", "total_cash_balance")
        buying_power = self.__value(
            summary,
            "buying_power",
            "BuyingPower",
            "margin_available_current",
        )

        if cash is not None:
            account.cash = Wallet(Amount(USD, float(cash)))
        if buying_power is not None:
            account.buying_power = Amount(USD, float(buying_power))

        account.portfolio = self._sync_positions()
        account.orders = self._sync_orders()
        return account

    def _cancel_order(self, order: Order) -> None:
        self.__client.cancel_order(order.id)

    def _update_order(self, order: Order) -> None:
        # SaxoClient exposes intent-based placement, not a generic replace call.
        self.__client.cancel_order(order.id)
        self._place_order(order)

    def _place_order(self, order: Order) -> None:
        kwargs = self.__order_kwargs(order)

        if order.is_mkt_order:
            self.__client.open_market(**kwargs)
        else:
            kwargs["order_price"] = order.limit
            self.__client.open_limit(**kwargs)
