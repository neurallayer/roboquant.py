from decimal import Decimal
import logging
from time import sleep
from typing import Any
from roboquant.brokers.broker import LiveBroker
import roboquant as rq
from roboquant.ibkr.types import AccountInfo, ContractInfo, PositionInfo


from ibind import IbkrClient, StockQuery, OrderRequest, QuestionType  # noqa: E402


logger = logging.getLogger(__name__)


class _AssetMapper:
    """Takes care of mapping between roboquant assets and IBKR contract-ids.
    It will cache previous results, so only for new assets or contracts it will make
    an API call.
    """

    def __init__(self, client: IbkrClient) -> None:
        self._asset_2_conid: dict[rq.Asset, int] = {}
        self._conid_2_asset: dict[int, rq.Asset] = {}
        self.client = client

    def update_mapping(self, asset: rq.Asset, conid: int):
        logger.debug("updating mapping asset=%s conid=%s", asset, conid)
        self._conid_2_asset[conid] = asset
        self._asset_2_conid[asset] = conid

    def get_conid(self, asset: rq.Asset) -> int | None:
        # First, check the mapping table
        if conid := self._asset_2_conid.get(asset):
            return conid

        if not isinstance(asset, rq.Stock):
            logger.warning("only stocks are supported, got %s", asset)
            return None

        contract_filter = {"isUS": True} if asset.currency == rq.monetary.USD else  {"isUS": False}

        query = StockQuery(asset.symbol, contract_conditions=contract_filter)
        data : dict = self.client.security_stocks_by_symbol([query], default_filtering=False).data  # type: ignore
        if contract_info := data.get(asset.symbol):
            if len(contract_info) == 1 and len(contract_info[0]["contracts"]) == 1:
                conid = contract_info[0]["contracts"][0]["conid"]
                logger.info("converted asset=%s into conid=%s", asset, conid)
                self.update_mapping(asset, conid)
                return conid

        logger.warning("couldn't determine conid for asset %s, result was %d", asset, data)

    def get_asset(self, conid: int) -> rq.Asset | None:
        if asset := self._conid_2_asset.get(int(conid)):
            return asset

        contract: ContractInfo = self.client.contract_information_by_conid(conid).data  # type: ignore

        match contract["instrument_type"]:
            case "STK":
                asset = rq.Stock(contract["symbol"], rq.monetary.Currency(contract["currency"]))
            case "OPT":
                asset = rq.Option(contract["local_symbol"], rq.monetary.Currency(contract["currency"]))
            case _:
                logger.warning("unsupported asset class %s", contract["instrument_type"])

        if asset:
            logger.info("converted contract=%s into asset=%s", contract, asset)
            self.update_mapping(asset, conid)
        else:
            logger.warning("couldn't find asset for conid %s", conid)

        return asset

# Default answers to questions asked by IBKR when placing orders.
default_answers = {
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
    """

    def __init__(self, account_id: str | None = None, ibkr_client: IbkrClient | None = None):
        """Initialize the IBKRBroker instance.
        Args:
            account_id (str | None): The account ID to use. If None, the first available account will be used.
            ibkr_client (IbkrClient | None): An optional IBKR client instance. If None, a new client will be created.
        """

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
        self.base_currency = rq.monetary.Currency(account_summary["currency"]) # type: ignore
        logger.info(f"using account={account_id} with base-currency={self.base_currency}")

        # We also need to call this once before using other order related calls
        client.live_orders()
        sleep(1)
        self.client = client
        self._mapper = _AssetMapper(client)


    def __get_positions(self) -> dict[rq.Asset, rq.Position]:
        """Return all the open positions"""
        result: dict[rq.Asset, rq.Position] = {}
        positions: list[PositionInfo] = self.client.positions().data or []  # type: ignore
        for pos_info in positions:
            conid = pos_info["conid"]
            if asset := self._mapper.get_asset(conid):
                if size := pos_info["position"]:
                    position = rq.Position(Decimal(size), pos_info["avgPrice"], pos_info["mktPrice"])
                    result[asset] = position
            else:
                logger.warning("ignoring position %s because couldn't map conid to asset", pos_info)
        return result

    def __get_orders(self) -> list[rq.Order]:
        result: dict[Any, rq.Order] = {}
        closed_status = {"Cancelled", "Filled", "Rejected", "Inactive"}
        orders = self.client.live_orders(force=False).data["orders"]  # type: ignore
        for order in orders:
            if order["orderType"] != "Limit":
                logger.warning("ignoring order that is not a limit order %s", order)
                continue
            conid = order["conid"]
            if asset := self._mapper.get_asset(conid):
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

    def __get_cash_bp(self):
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

    def __create_order_request(self, order: rq.Order) -> OrderRequest:
        conid = self._mapper.get_conid(order.asset)
        if conid is None:
            raise ValueError(f"Cannot determine contract-id for asset {order.asset}")

        qty = float(abs(order.size))
        side = "BUY" if order.size > 0 else "SELL"
        return OrderRequest(
            conid=int(conid),
            side=side,
            quantity=qty,
            order_type="LMT",
            acct_id=str(self.client.account_id),
            price=order.limit,
            tif=order.tif,
            **order.info
        )

    def _update_order(self, order: rq.Order):
        req = self.__create_order_request(order)
        result = self.client.modify_order(order.id, req, answers=default_answers)
        logger.info("update order result %s", result)

    def _place_order(self, order: rq.Order):
        req = self.__create_order_request(order)
        result = self.client.place_order(req, answers=default_answers)
        logger.info("place order result %s", result)

    def _cancel_order(self, order: rq.Order):
        result = self.client.cancel_order(order.id)
        logger.info("cancel order result %s", result)

    def _get_account(self):
        account = rq.account.Account()
        account.positions = self.__get_positions()
        account.orders = self.__get_orders()
        cash, bp = self.__get_cash_bp()
        account.last_update = rq.utcnow()
        account.cash[self.base_currency] = cash
        account.buying_power = rq.Amount(self.base_currency, bp)
        return account
