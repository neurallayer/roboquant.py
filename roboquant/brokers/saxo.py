from __future__ import annotations

import os
from typing import Any, Mapping, override

import requests

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio


class SaxoBroker(LiveBroker):
    """Broker implementation using Saxo OpenAPI."""

    def __init__(
        self,
        access_token: str | None = None,
        account_key: str | None = None,
        client_key: str | None = None,
        base_url: str | None = None,
    ):
        self._access_token = access_token or os.environ["SAXO_ACCESS_TOKEN"]
        self._account_key = account_key or os.environ["SAXO_ACCOUNT_KEY"]
        self._client_key = client_key or os.getenv("SAXO_CLIENT_KEY")
        self._base_url = (
            base_url
            or os.getenv("SAXO_BASE_URL")
            or "https://gateway.saxobank.com/openapi"
        ).rstrip("/")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        query = {"AccountKey": self._account_key}
        if self._client_key:
            query["ClientKey"] = self._client_key
        if params:
            query.update(params)

        response = self._session.request(
            method,
            f"{self._base_url}{path}",
            params=query,
            json=json,
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else None

    @staticmethod
    def _value(value: Any) -> Any:
        return getattr(value, "value", value)

    def __get_portfolio(self) -> Portfolio:
        """Get open net positions."""
        data = self._request(
            "GET",
            "/port/v1/netpositions/me",
            params={
                "FieldGroups": "DisplayAndFormat,NetPositionBase,NetPositionView"
            },
        )

        portfolio = Portfolio()
        positions = getattr(portfolio, "positions", None)

        for item in data.get("Data", []):
            view = item.get("NetPositionView", {})
            base = item.get("NetPositionBase", {})
            instrument = (
                view.get("DisplayAndFormat", {}).get("Symbol")
                or base.get("DisplayAndFormat", {}).get("Symbol")
                or item.get("Uic")
            )
            quantity = view.get("CurrentPriceDelayMinutes")
            quantity = (
                view.get("Amount")
                or view.get("NetPositionAmount")
                or base.get("Amount")
                or item.get("Amount")
            )

            if instrument is not None and quantity is not None:
                if isinstance(positions, dict):
                    positions[instrument] = quantity
                else:
                    setattr(portfolio, "positions", {instrument: quantity})

        return portfolio

    def __get_orders(self) -> list[Order]:
        """Get open orders."""
        data = self._request(
            "GET",
            "/port/v1/orders/me",
            params={
                "FieldGroups": (
                    "DisplayAndFormat,ExchangeInfo,OrderDetails,"
                    "OrderRelatedData,TradingSchedule"
                )
            },
        )

        orders: list[Order] = []
        for item in data.get("Data", []):
            details = item.get("OrderDetails", {})
            display = item.get("DisplayAndFormat", {})

            order = Order(
                asset = display.get("Symbol") or item.get("Uic"),
                size = item.get("Amount"),
                id = item.get("OrderId"),
                limit = details.get("OrderPrice") or item.get("OrderPrice")
            )
            orders.append(order)

        return orders

    @override
    def _get_account(self) -> Account:
        account = Account()
        account.orders = self.__get_orders()
        account.portfolio = self.__get_portfolio()
        return account

    @staticmethod
    def _order_id(order: Order) -> str:
        order_id = (
            getattr(order, "order_id", None)
            or getattr(order, "id", None)
            or getattr(order, "broker_order_id", None)
        )
        if not order_id:
            raise ValueError("The order does not have a Saxo order ID")
        return str(order_id)

    def _order_payload(self, order: Order) -> dict[str, Any]:
        instrument = getattr(order, "instrument", None)
        uic = (
            getattr(order, "uic", None)
            or getattr(instrument, "uic", None)
            or getattr(instrument, "id", None)
        )
        if uic is None:
            raise ValueError("The order instrument must provide a Saxo UIC")

        side = str(self._value(getattr(order, "side", ""))).upper()
        order_type = str(
            self._value(getattr(order, "order_type", "Market"))
        ).replace("_", "").lower()

        type_map = {
            "market": "Market",
            "limit": "Limit",
            "stop": "Stop",
            "stoplimit": "StopLimit",
        }

        payload: dict[str, Any] = {
            "AccountKey": self._account_key,
            "AssetType": getattr(instrument, "asset_type", "Stock"),
            "Uic": uic,
            "BuySell": "Buy" if side in {"BUY", "B", "1"} else "Sell",
            "Amount": abs(getattr(order, "quantity")),
            "OrderType": type_map.get(order_type, "Market"),
            "ManualOrder": True,
        }

        limit_price = getattr(order, "limit_price", None)
        if limit_price is not None:
            payload["OrderPrice"] = limit_price

        stop_price = getattr(order, "stop_price", None)
        if stop_price is not None:
            payload["StopLimitPrice"] = stop_price

        return payload

    @override
    def _cancel_order(self, order: Order):
        """Cancel an open order."""
        self._request("DELETE", f"/trade/v2/orders/{self._order_id(order)}")

    @override
    def _update_order(self, order: Order):
        """Modify an existing order."""
        payload = self._order_payload(order)
        payload["OrderId"] = self._order_id(order)
        self._request("PATCH", "/trade/v2/orders", json=payload)

    @override
    def _place_order(self, order: Order):
        """Place a single market or limit order."""
        response = self._request(
            "POST",
            "/trade/v2/orders",
            json=self._order_payload(order),
        )

        if isinstance(response, dict):
            order_id = response.get("OrderId") or response.get("OrderIdTrailing")
            print(order_id)