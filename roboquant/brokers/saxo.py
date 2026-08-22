from __future__ import annotations

from decimal import Decimal
import os
from typing import Any, Mapping, override

import requests

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset, Stock
from roboquant.common.monetary import Amount, Currency, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position


class SaxoBroker(LiveBroker):
    """Broker implementation using Saxo OpenAPI."""

    def __init__(
        self,
        access_token: str | None = None,
        account_key: str | None = None,
        client_key: str | None = None,
    ):
        super().__init__()
        self._access_token = access_token or os.environ["SAXO_ACCESS_TOKEN"]
        self._account_key = None
        self._client_key = None
        self._asset_mapping: dict[Asset, tuple[int, str]] = {}

        self._base_url = "https://gateway.saxobank.com/sim/openapi"

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        default_client_key, default_acc_key = self.__get_defaults()
        self._account_key = account_key or os.getenv("SAXO_ACCOUNT_KEY") or default_acc_key
        self._client_key = client_key or os.getenv("SAXO_CLIENT_KEY") or default_client_key
        self._last_prices = {}


    def __get_asset(self, uic: int, assetType: str) -> Asset:
        for k, v in self._asset_mapping.items():
            if v[0] == uic and v[1] == assetType:
                return k
        data = self._request("GET", f"/ref/v1/instruments/details/{uic}/{assetType}")
        symbol = data["Symbol"].split(":")[0]
        asset = Stock(symbol, Currency(data["CurrencyCode"]))
        self._asset_mapping[asset] = (uic, assetType)
        return asset

    def __get_defaults(self):
        data = self._request("GET", "/port/v1/clients/me")
        return data["ClientKey"], data["DefaultAccountKey"]

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
        include_account: bool = True,
        include_client: bool = True,
    ) -> Any:
        query = {}
        if self._account_key and include_account:
            query["AccountKey"] = self._account_key
        if self._client_key and include_client:
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


    def __get_price(self, asset: Asset, price_type:str="Close") -> float:
        if price := self._last_prices.get(asset):
            return price
        uic, asset_type = self._asset_mapping[asset]
        data = self._request(
                "GET",
                "/chart/v3/charts",
                params={
                    "AssetType": asset_type,
                    "Uic" : uic,
                    "Count" : 1,
                    "Horizon": 1
                },
            )
        price = data["Data"][0][price_type]
        self._last_prices[asset] = price
        return price


    def __get_portfolio(self) -> Portfolio:
        """Get open net positions."""
        data = self._request(
            "GET",
            "/port/v1/netpositions/me",
            params={
                "FieldGroups": "NetPositionBase,NetPositionView"
            },
        )

        portfolio = Portfolio()

        for item in data.get("Data", []):
            view = item.get("NetPositionView", {})
            base = item.get("NetPositionBase", {})

            asset = self.__get_asset(base["Uic"], base["AssetType"])
            size = Decimal(base["Amount"])
            avg_price = view["AverageOpenPrice"]
            mkt_price = view["CurrentPrice"] or self.__get_price(asset)
            portfolio[asset] = Position(size, avg_price, mkt_price)

        return portfolio

    def __get_orders(self) -> list[Order]:
        """Get open orders."""
        data = self._request(
            "GET",
            "/port/v1/orders",
            params={
                "FieldGroups": (
                    "DisplayAndFormat,ExchangeInfo"
                )
            },
        )

        orders: list[Order] = []
        for item in data.get("Data", []):

            asset = self.__get_asset(item["Uic"], item["AssetType"])
            assert asset
            is_mkt = item["OpenOrderType"] == "Market"
            is_buy = item["BuySell"] == "Buy"
            order = Order(
                asset = asset,
                size = Decimal(item.get("Amount")) if is_buy else -Decimal(item.get("Amount")),
                id = item.get("OrderId"),
                limit = None if is_mkt else item["Price"]
            )
            orders.append(order)

        return orders


    def __get_base_account(self):
        data = self._request(
            "GET",
            "/port/v1/balances",
            )
        base_currency = Currency(data["Currency"])
        cash = Amount(base_currency, data["CashBalance"])
        bp = Amount(base_currency, data["CashAvailableForTrading"])
        acc = Account(base_currency)
        acc.cash = Wallet(cash)
        acc.buying_power = bp
        return acc

    @override
    def _get_account(self) -> Account:
        account = self.__get_base_account()
        account.portfolio = self.__get_portfolio()
        account.orders = self.__get_orders()
        return account

    def _order_payload(self, order: Order) -> dict[str, Any]:
        uic, assetType = self._asset_mapping[order.asset]
        if uic is None:
            raise ValueError("The order instrument must provide a Saxo UIC")

        payload: dict[str, Any] = {
            "AccountKey": self._account_key,
            "AssetType": assetType,
            "Uic": uic,
            "BuySell": "Buy" if order.is_buy else "Sell",
            "Amount": str(abs(order.size)),
            "OrderType": "Market" if order.is_mkt_order else "Limit",
            "ManualOrder": True,
        }

        limit_price = order.limit
        if limit_price is not None:
            payload["OrderPrice"] = limit_price

        return payload

    @override
    def _cancel_order(self, order: Order):
        """Cancel an open order."""
        self._request("DELETE", f"/trade/v2/orders/{order.id}")

    @override
    def _update_order(self, order: Order):
        """Modify an existing order."""
        payload = self._order_payload(order)
        payload["OrderId"] = order.id
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
