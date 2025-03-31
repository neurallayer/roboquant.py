from decimal import Decimal
import logging
from time import sleep
from typing import Any, TypedDict
from roboquant.brokers.broker import LiveBroker
import roboquant as rq


from ibind import IbkrClient, StockQuery, OrderRequest, QuestionType  # noqa: E402


logger = logging.getLogger(__name__)

AccountInfo = TypedDict(
    "AccountInfo",
    {
        "id": str,
        "PrepaidCrypto-Z": bool,
        "PrepaidCrypto-P": bool,
        "brokerageAccess": bool,
        "accountId": str,
        "accountVan": str,
        "accountTitle": str,
        "displayName": str,
        "accountAlias": str | None,
        "accountStatus": int,
        "currency": str,
        "type": str,
        "tradingType": str,
        "businessType": str,
        "category": str,
        "ibEntity": str,
        "faclient": bool,
        "clearingStatus": str,
        "covestor": bool,
        "noClientTrading": bool,
        "trackVirtualFXPortfolio": bool,
        "acctCustType": str,
        "parent": dict,
        "desc": str,
    },
    total=True,
)


PositionInfo = TypedDict(
    "PositionInfo",
    {
        "acctId": str,
        "conid": int,
        "contractDesc": str,
        "position": float,
        "mktPrice": float,
        "mktValue": float,
        "currency": str,
        "avgCost": float,
        "avgPrice": float,
        "realizedPnl": float,
        "unrealizedPnl": float,
        "exchs": str | None,
        "expiry": str | None,
        "putOrCall": str | None,
        "multiplier": float,
        "strike": str,
        "exerciseStyle": str | None,
        "conExchMap": list,
        "assetClass": str,
        "undConid": int,
        "model": str,
        "baseMktValue": float,
        "baseMktPrice": float,
        "baseAvgCost": float,
        "baseAvgPrice": float,
        "baseRealizedPnl": float,
        "baseUnrealizedPnl": float,
        "incrementRules": list,
        "displayRule": dict,
        "time": int,
        "chineseName": str,
        "allExchanges": str,
        "listingExchange": str,
        "countryCode": str,
        "name": str,
        "lastTradingDay": str | None,
        "group": str,
        "sector": str,
        "sectorGroup": str,
        "ticker": str,
        "type": str,
        "hasOptions": bool,
        "fullName": str,
        "isUS": bool,
        "isEventContract": bool,
        "pageSize": int,
    },
    total=True,
)


ContractInfo = TypedDict(
    "ContractInfo",
    {
        "cfi_code": str,
        "symbol": str,
        "cusip": str | None,
        "expiry_full": str | None,
        "con_id": int,
        "maturity_date": str | None,
        "industry": str,
        "instrument_type": str,
        "trading_class": str,
        "valid_exchanges": str,
        "allow_sell_long": bool,
        "is_zero_commission_security": bool,
        "local_symbol": str,
        "contract_clarification_type": str | None,
        "classifier": str | None,
        "currency": str,
        "text": str | None,
        "underlying_con_id": int,
        "r_t_h": bool,
        "multiplier": str | None,
        "underlying_issuer": str | None,
        "contract_month": str | None,
        "company_name": str,
        "smart_available": bool,
        "exchange": str,
        "category": str,
    },
    total=True,
)


_asset_2_conid: dict[rq.Asset, int] = {}
_conid_2_asset: dict[int, rq.Asset] = {}


def _update_cache(asset: rq.Asset, conid: int):
    _conid_2_asset[conid] = asset
    _asset_2_conid[asset] = conid


answers = {
    QuestionType.PRICE_PERCENTAGE_CONSTRAINT: True,
    QuestionType.ORDER_VALUE_LIMIT: True,
    QuestionType.MISSING_MARKET_DATA: True,
    QuestionType.STOP_ORDER_RISKS: True,
    "exceeds the Size Limit": True,
}


class IBKRBroker(LiveBroker):
    """Broker implementation for Interactive Brokers using the IBKR Web API.
    This class provides an interface to interact with Interactive Brokers (IBKR) for live trading.
    It supports operations such as retrieving account information, managing positions, placing
    and modifying orders, and fetching live orders. Currently, it only supports stocks.

    Attributes:
        base_currency (rq.monetary.Currency): The base currency of the account.
        client (IbkrClient): The IBKR client used for API communication.
    Methods:
        __init__(account_id: str | None = None):
            Initializes the IBKRBroker instance and sets up the client connection.
        _get_positions() -> dict[rq.Asset, rq.Position]:
            Retrieves the current positions in the account.
        _get_orders() -> list[rq.Order]:
            Retrieves the list of active orders in the account.
        _update_order(order: rq.Order):
            Updates an existing order in IBKR.
        _place_order(order: rq.Order):
            Places a new order in IBKR.
        _cancel_order(order: rq.Order):
            Cancels an existing order in IBKR.
        _get_account() -> rq.account.Account:
            Retrieves the account information, including positions, orders, cash, and buying power.
        _get_cash_bp() -> tuple[float, float]:
            Retrieves the cash balance and buying power of the account.
    """

    def __init__(self, account_id: str | None = None, ibkr_client: IbkrClient | None = None):
        super().__init__()
        client = ibkr_client or IbkrClient()
        ok = client.check_health()
        assert ok, "health not ok"
        sleep(1)

        accounts = {}
        while not accounts:
            accounts = client.receive_brokerage_accounts().data
            sleep(1)

        account_id = account_id or accounts["accounts"][0]  # type: ignore
        client.account_id = account_id
        account_summary: AccountInfo = client.portfolio_account_information().data  # type: ignore
        self.base_currency = rq.monetary.Currency(account_summary["currency"])
        logger.info(f"using account={account_id} with base-currency={self.base_currency}")

        # We also need to call this once before using other code
        client.live_orders()
        sleep(1)
        self.client = client

    def __find_conid(self, asset: rq.Asset) -> int | None:
        if conid := _asset_2_conid.get(asset):
            return conid

        if asset.currency == rq.monetary.USD:
            filter = {"isUS": True}
        else:
            filter = {"isUS": False}

        query = StockQuery(asset.symbol, contract_conditions=filter)

        data = self.client.security_stocks_by_symbol([query], default_filtering=False).data
        if data and len(data) == 1:
            conid = data[0]["conid"]
            logger.info("converted asset=%s into conid=%s", asset, conid)
            _update_cache(asset, conid)
            return conid

        logger.warning("couldn't determine conid for asset %s", asset)

    def __get_asset(self, conid: int) -> rq.Asset | None:
        if asset := _conid_2_asset.get(int(conid)):
            return asset

        contract: ContractInfo = self.client.contract_information_by_conid(conid).data  # type: ignore
        match contract["instrument_type"]:
            case "STK":
                asset = rq.Stock(contract["symbol"], rq.monetary.Currency(contract["currency"]))
            case _:
                logger.warning("unsupported asset class %s", contract["instrument_type"])

        if asset:
            logger.info("converted contract=%s into asset=%s", contract, asset)
            _update_cache(asset, conid)
        else:
            logger.warning("could create asset for conid %s", conid)

        return asset

    def _get_positions(self) -> dict[rq.Asset, rq.Position]:
        """Return all the open positions"""
        result: dict[rq.Asset, rq.Position] = {}
        positions: list[PositionInfo] = self.client.positions().data or []  # type: ignore
        for position in positions:
            conid = position["conid"]
            if asset := self.__get_asset(conid):
                if size := position["position"]:
                    p = rq.Position(Decimal(size), position["avgPrice"], position["mktPrice"])
                    result[asset] = p
            else:
                logger.warning("ignoring position %s because couldn't map conid to asset", position)
        return result

    def _get_orders(self) -> list[rq.Order]:
        result: dict[Any, rq.Order] = {}
        closed_status = {"Cancelled", "Filled", "Rejected", "Inactive"}
        orders = self.client.live_orders(force=False).data["orders"]  # type: ignore
        for order in orders:
            if order["orderType"] != "Limit":
                logger.warning("ignoring order that is not a limit order %s", order)
                continue
            conid = order["conid"]
            if asset := self.__get_asset(conid):
                if order["status"] in closed_status:
                    logger.info("ignoring closed order %s", order)
                    continue
                size = order["totalSize"]
                if order["side"] == "SELL":
                    size = -size
                new_order = rq.Order(asset, Decimal(size), float(order["price"]))
                new_order.id = order["orderId"]
                fill = order["filledQuantity"]
                if order["side"] == "SELL":
                    fill = -fill
                new_order.fill = Decimal(fill)

                if "cancel" in order["order_ccp_status"]:
                    new_order.size = Decimal()

                # store the latest order info found for an id
                result[new_order.id] = new_order

        return list(result.values())

    def __get_order_request(self, order: rq.Order) -> OrderRequest:
        extra_info = {"outside_rth"}
        kwargs = {k: v for k, v in order.info.items() if k in extra_info}

        conid = self.__find_conid(order.asset)
        if conid is None:
            raise ValueError(f"Cannot determine contract-id for asset {order.asset}")

        qty = float(abs(order.size))
        side = "BUY" if order.size > 0 else "SELL"
        req = OrderRequest(
            conid=int(conid),
            side=side,
            quantity=qty,
            order_type="LMT",
            acct_id=str(self.client.account_id),
            price=order.limit,
            **kwargs,
        )
        return req

    def _update_order(self, order: rq.Order):
        assert order.id, "no known order id"
        req = self.__get_order_request(order)
        result = self.client.modify_order(order.id, req, answers=answers)  # type: ignore
        logger.info("update order result %s", result)

    def _place_order(self, order: rq.Order):
        assert not order.id, "cannot place an existing order"
        req = self.__get_order_request(order)
        result = self.client.place_order(req, answers=answers)  # type: ignore
        logger.info("place order result %s", result)

    def _cancel_order(self, order: rq.Order):
        assert order.id, "cancel order needs an id"
        result = self.client.cancel_order(order.id)
        logger.info("cancel order result %s", result)

    def _get_account(self):
        account = rq.account.Account()
        account.positions = self._get_positions()
        account.orders = self._get_orders()
        cash, bp = self._get_cash_bp()
        account.last_update = rq.utcnow()
        account.cash[self.base_currency] = cash
        account.buying_power = rq.Amount(self.base_currency, bp)
        return account

    def _get_cash_bp(self):
        """Get the total cash balance"""
        summary: dict = self.client.account_summary().data  # type: ignore
        bp = summary["buyingPower"]
        cash = 0.0
        balances: list = summary["cashBalances"]
        for balance in balances:
            ccy: str = balance["currency"]
            if ccy.startswith("Total"):
                cash = balance["balance"]
                break
        return cash, bp
