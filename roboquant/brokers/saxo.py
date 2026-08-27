from dataclasses import replace
from decimal import Decimal
import logging
import json
import importlib.resources
from typing import Any, Mapping, override

import requests

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset, Forex, Stock
from roboquant.common.monetary import Amount, Currency, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position

from roboquant.brokers._saxo_types import NetPositionsResponse, OpenOrdersResponse

logger = logging.getLogger(__name__)

class SaxoBroker(LiveBroker):
    """Saxo Broker implementation.
    This implementation use the Saxo OpenAPI (REST) directly without requiring an additional
    specific Saxo client library."""

    def __init__(
        self,
        access_token: str,
        account_key: str | None = None,
        client_key: str | None = None,
        simulator: bool = True
    ):
        super().__init__()
        self._access_token : str = access_token
        self._account_key : str | None= None
        self._client_key : str | None = None
        self._asset_mapping: dict[Asset, tuple[int, str]] = {}

        assert simulator, "right now only simulator mode is supported"
        self._base_url = "https://gateway.saxobank.com/sim/openapi" if simulator else "https://gateway.saxobank.com/openapi"

        self._session: requests.Session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        default_client_key, default_acc_key = self.__get_defaults()
        self._account_key = account_key or default_acc_key
        self._client_key = client_key or default_client_key
        self._last_prices = {}
        self._load_assets()
        self.__simplify_assets()

    def reset_session(self):
        """Close the old session and start a new one"""
        try:
            self._session.close()
        except: # noqa: E722
            pass
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def __get_asset(self, uic: int, assetType: str) -> Asset:
        for k, v in self._asset_mapping.items():
            if v[0] == uic and v[1] == assetType:
                return k
        data = self.__request("GET", f"/ref/v1/instruments/details/{uic}/{assetType}")
        symbol = data["Symbol"]
        asset = Stock(symbol, Currency(data["CurrencyCode"]))
        self._asset_mapping[asset] = (uic, assetType)
        return asset

    def __get_defaults(self) -> tuple[str, str]:
        data = self.__request("GET", "/port/v1/clients/me")
        return data["ClientKey"], data["DefaultAccountKey"]

    def assets(self) -> list[Asset]:
        """Return the current known assets with this broker"""
        return list(self._asset_mapping.keys())

    def _refresh_all_stocks(self):
        def get_relevant_data(result: dict[str, Any]):
            keys = ["Symbol", "CurrencyCode", "Identifier", "AssetType"]
            r = []
            for row in result["Data"]:
                r.append([row[k] for k in keys ])
            return r

        data = []
        result = self.__request(
            "GET",
            "/ref/v1/instruments",
            params={"AssetTypes": "Stock, FxSpot, Etf, Fund", "$top" : 500}
        )
        data.extend(get_relevant_data(result))

        while "__next" in result:
            print(".", end="", flush=True)
            response = self._session.request("GET", result["__next"], timeout=30)
            result = response.json()
            if "Data" in result:
                data.extend(get_relevant_data(result))
            else:
                break

        with open('__saxo_assets.json', 'w') as f:
            json.dump(data, f)

    def _load_assets(self):
        """Load assets from a included file"""
        json_str = importlib.resources.read_text(self.__module__, "__saxo_assets.json")
        data: list[tuple[str,str,int,str]] = json.loads(json_str)
        for row in data:
            symbol, currencyCode, uic, asset_type = row
            currency = Currency(currencyCode)
            match asset_type:
                case "Stock" | "Etf" | "Fund":
                    asset = Stock(symbol, currency)
                case "FxSpot":
                    asset = Forex(symbol, currency)
                case _:
                    logger.warning("unexpected asset type %s", asset_type)
                    continue
            if asset in self._asset_mapping:
                logger.warning("duplicate asset %s %s", asset, row)
            self._asset_mapping[asset] = (uic, asset_type)

        logger.info("loaded %s assets", len(self._asset_mapping))

    def __simplify_assets(self):
        """Use simpler symbol names if there is no clash.
        This makes it easier to map feed assets and order assets
        """
        tmp: set[Asset] = set()
        duplicate: set[Asset] = set()
        for asset in self._asset_mapping.keys():
            simplified_symbol = asset.symbol.split(":")[0]
            a = replace(asset, symbol = simplified_symbol)
            if a in tmp:
                duplicate.add(a)
            else:
                tmp.add(a)

        result: dict[Asset, tuple[int, str]] = {}
        for asset, v in self._asset_mapping.items():
            simplified_symbol = asset.symbol.split(":")[0]
            a = replace(asset, symbol = simplified_symbol)
            if a in duplicate:
                result[asset] = v
            else:
                result[a] = v

        assert len(result) == len(self._asset_mapping)
        self._asset_mapping = result

    def __request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None
    ) -> Any:
        query = {}
        if self._account_key:
            query["AccountKey"] = self._account_key
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

    def get_price(self, asset: Asset) -> float:
        """Get the latest known price for an asset using the Saxo charts API.
        It will cache the price and is mostly used to set mkt_price for positions
        when the market is closed.
        """
        if asset not in self._last_prices:
            uic, asset_type = self._asset_mapping[asset]
            data = self.__request(
                "GET",
                "/chart/v3/charts",
                params={"AssetType": asset_type, "Uic": uic, "Count": 1, "Horizon": 1},
            )
            price = data["Data"][0]["Close"]
            self._last_prices[asset] = price
        return self._last_prices[asset]

    def __get_portfolio(self) -> Portfolio:
        """Get open net positions."""
        data: NetPositionsResponse = self.__request(
            "GET",
            "/port/v1/netpositions/me",
            params={"FieldGroups": "NetPositionBase,NetPositionView"},
        )

        portfolio = Portfolio()

        for item in data.get("Data", []):
            view = item.get("NetPositionView", {})
            base = item.get("NetPositionBase", {})

            asset = self.__get_asset(base["Uic"], base["AssetType"])
            size = Decimal(base["Amount"])
            avg_price = view["AverageOpenPrice"]

            # Unfortunately Saxo doesn't include the latest market price if
            # the market is closed. So in those case we need to get the price.
            mkt_price = view["CurrentPrice"] or self.get_price(asset)
            pos = Position(asset, size, avg_price, mkt_price)
            portfolio.append(pos)

        return portfolio

    def __get_orders(self) -> list[Order]:
        """Get open orders."""
        data: OpenOrdersResponse = self.__request(
            "GET",
            "/port/v1/orders",
            params={"FieldGroups": ("DisplayAndFormat,ExchangeInfo")},
        )

        orders: list[Order] = []
        for item in data.get("Data", []):
            asset = self.__get_asset(item["Uic"], item["AssetType"])
            assert asset
            is_mkt = item["OpenOrderType"] == "Market"
            is_buy = item["BuySell"] == "Buy"
            tif = "GTC" if item["Duration"]["DurationType"] == "GoodTillCancel" else "DAY"
            order = Order(
                asset=asset,
                size=Decimal(item.get("Amount")) if is_buy else -Decimal(item.get("Amount")),
                id=item.get("OrderId"),
                limit=None if is_mkt else item["Price"],
                tif = tif
            )
            orders.append(order)

        return orders

    def __get_base_account(self):
        data = self.__request(
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

    def match_asset(self, asset: Asset) -> Asset | None:
        """Tries to match an asset, even if not 100% matching symbol name."""
        for key in self._asset_mapping.keys():
            short_symbol = key.symbol.split(":")[0]
            if short_symbol == asset.symbol and key.currency == asset.currency and key.asset_class == asset.asset_class:
                return key
        return None

    def __order_payload(self, order: Order) -> dict[str, Any]:
        uic, assetType = self._asset_mapping[order.asset]

        payload: dict[str, Any] = {
            "AccountKey": self._account_key,
            "AssetType": assetType,
            "Uic": uic,
            "BuySell": "Buy" if order.is_buy else "Sell",
            "Amount": str(abs(order.size)),
            "OrderType": "Market" if order.is_mkt_order else "Limit",
            "ManualOrder": True,
            "OrderDuration": {
                "DurationType": "GoodTillCancel" if order.tif == "GTC" else "DayOrder"
            },
        }

        if order.is_limit_order:
            payload["OrderPrice"] = order.limit

        return payload

    @override
    def _cancel_order(self, order: Order):
        """Cancel an open order."""
        resp = self.__request("DELETE", f"/trade/v2/orders/{order.id}")
        logger.info("cancelled order=%s resp=%s", order, resp)

    @override
    def _update_order(self, order: Order):
        """Modify an existing order."""
        payload = self.__order_payload(order)
        payload["OrderId"] = order.id
        resp = self.__request("PATCH", "/trade/v2/orders", json=payload)
        logger.info("updated order=%s resp=%s", order, resp)

    @override
    def _place_order(self, order: Order):
        """Place a single market or limit order."""
        resp = self.__request(
            "POST",
            "/trade/v2/orders",
            json=self.__order_payload(order),
        )

        logger.info("placed order=%s resp=%s", order, resp)
