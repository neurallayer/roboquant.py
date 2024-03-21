import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from ibapi import VERSION
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order as IBKROrder
from ibapi.wrapper import EWrapper

from roboquant.account import Account, Position
from roboquant.event import Event
from roboquant.order import Order, OrderStatus
from roboquant.brokers.broker import Broker, _update_positions

assert VERSION["major"] == 10 and VERSION["minor"] == 19, "Wrong version of the IBAPI found"

logger = logging.getLogger(__name__)


# noinspection PyPep8Naming
class _IBApi(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.__account = {AccountSummaryTags.TotalCashValue: 0.0, AccountSummaryTags.BuyingPower: 0.0}
        self.__account_end = threading.Condition()
        self.__order_id = 0

    def nextValidId(self, orderId: int):
        self.__order_id = orderId
        logger.debug("The next valid order id is: %s", orderId)

    def get_next_order_id(self):
        result = str(self.__order_id)
        self.__order_id += 1
        return result

    def position(self, account: str, contract: Contract, position: Decimal, avgCost: float):
        logger.debug("position=%s symbol=%s  avgCost=%s", position, contract.localSymbol, avgCost)
        symbol = contract.localSymbol or contract.symbol
        old_position = self.positions.get(symbol)
        mkt_price = old_position.mkt_price if old_position else avgCost
        self.positions[symbol] = Position(position, avgCost, mkt_price)

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        logger.debug("account %s=%s", tag, value)
        self.__account[tag] = float(value)

    def accountSummaryEnd(self, reqId: int):
        with self.__account_end:
            self.__account_end.notify_all()

    def openOrder(self, orderId: int, contract, order: IBKROrder, orderState):
        logger.debug(
            "openOrder orderId=%s status=%s size=%s limit=%s",
            orderId,
            orderState.status,
            order.totalQuantity,
            order.lmtPrice,
        )
        size = order.totalQuantity if order.action == "BUY" else -order.totalQuantity
        symbol = contract.localSymbol
        rq_order = Order(symbol, size) if not order.lmtPrice else Order(symbol, size, order.lmtPrice)
        rq_order.id = str(orderId)
        self.orders[rq_order.id] = rq_order

    def request_account(self):
        """blocking call till account summary has been received"""
        buyingpower_tag = AccountSummaryTags.BuyingPower
        cash_tag = AccountSummaryTags.TotalCashValue
        with self.__account_end:
            super().reqAccountSummary(1, "All", f"{buyingpower_tag},{cash_tag}")
            self.__account_end.wait()

    def get_buying_power(self):
        buyingpower_tag = AccountSummaryTags.BuyingPower
        return self.__account[buyingpower_tag] or 0.0

    def get_cash(self):
        cash_tag = AccountSummaryTags.TotalCashValue
        return self.__account[cash_tag] or 0.0

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice,
    ):
        logger.debug("order status orderId=%s status=%s fill=%s", orderId, status, filled)
        orderId = str(orderId)
        if orderId in self.orders:
            order = self.orders[orderId]
            order.fill = filled
            match status:
                case "Submitted":
                    order.status = OrderStatus.ACTIVE
                case "Cancelled":
                    order.status = OrderStatus.CANCELLED
                case "Filled":
                    order.status = OrderStatus.FILLED
        else:
            logger.warning("received status for unknown order id=%s status=%s", orderId, status)


class IBKRBroker(Broker):
    """
    Attributes
    ==========
    contract_mapping
        Map symbols to IBKR contracts.
        If a symbol is not found, the symbol is assumed to represent a US stock

    host
        the ip number of the host where TWS or IB Gateway is running.

    port
       By default, TWS uses socket port 7496 for live sessions and 7497 for paper sessions.
       IB Gateway by contrast uses 4001 for live sessions and 4002 for paper sessions.
       However these are just defaults, and can be modified as desired.

    client_id
        The client id to use to connect to TWS or IB Gateway.
    """

    def __init__(self, host="127.0.0.1", port=4002, client_id=123) -> None:
        self.__account = Account()
        self.contract_mapping: dict[str, Contract] = {}
        api = _IBApi()
        api.connect(host, port, client_id)
        self.__api = api
        self.__has_new_orders_since_sync = False
        self.price_type = "DEFAULT"
        self.sleep_after_cancel = 0.0

        # Start the handling in a thread
        self.__api_thread = threading.Thread(target=api.run, daemon=True)
        self.__api_thread.start()
        time.sleep(3.0)

    @classmethod
    def use_tws(cls, client_id=123):
        """Return a broker connected to the TWS papertrade instance with its default port (7497) settings"""
        return cls("127.0.0.1", 7497, client_id)

    @classmethod
    def use_ibgateway(cls, client_id=123):
        """Return a broker connected to a IB Gateway papertrade instance with its default port (4002) settings"""
        return cls("127.0.0.1", 4002, client_id)

    def disconnect(self):
        self.__api.reader.conn.disconnect()  # type: ignore

    def _should_sync(self, now: datetime):
        """Avoid too many API calls"""
        return self.__has_new_orders_since_sync or now - self.__account.last_update > timedelta(seconds=1)

    def sync(self, event: Event | None = None) -> Account:
        """Sync with the IBKR account"""

        logger.debug("start sync")
        now = datetime.now(timezone.utc)

        if event:
            # Let make sure we don't use IBKRBroker by mistake during a back-test.
            if now - event.time > timedelta(minutes=30):
                logger.critical("received event from the past, now=%s event-time=%s", now, event.time)
                raise ValueError(f"received event too far in the past now={now} event-time={event.time}")

        api = self.__api
        acc = self.__account
        if self._should_sync(now):
            acc.last_update = now
            self.__has_new_orders_since_sync = False

            api.reqPositions()
            api.reqOpenOrders()
            api.request_account()

            acc.positions = {k: v for k, v in api.positions.items() if not v.size.is_zero()}
            acc.orders = list(api.orders.values())
            acc.buying_power = api.get_buying_power()
            acc.cash = api.get_cash()

        _update_positions(acc, event)
        logger.debug("end sync")
        return acc

    def place_orders(self, orders):

        self.__has_new_orders_since_sync = len(orders) > 0

        for idx, order in enumerate(orders, start=1):
            if idx % 25 == 0:
                # avoid to many API calls
                time.sleep(1)

            assert order.is_open, "can only place open orders"
            if order.size.is_zero():
                assert order.id is not None, "can only cancel orders with an id"
                self.__api.cancelOrder(int(order.id), "")
                if self.sleep_after_cancel:
                    time.sleep(self.sleep_after_cancel)
            else:
                if order.id is None:
                    order.id = self.__api.get_next_order_id()
                    self.__api.orders[order.id] = order
                ibkr_order = self._get_order(order)
                contract = self._get_contract(order)
                self.__api.placeOrder(int(order.id), contract, ibkr_order)

    @staticmethod
    def __update_ibkr_object(obj, update):
        if not update:
            return
        assert isinstance(update, dict)
        for name, value in update.items():
            if hasattr(obj, name):
                setattr(obj, name, value)
            else:
                logger.warning("unknown field name=%s value=%s", name, value)

    def _get_contract(self, order: Order) -> Contract:
        """Map an order to a IBKR contract."""

        c = self.contract_mapping.get(order.symbol)

        if not c:
            c = Contract()
            c.symbol = order.symbol
            c.secType = "STK"
            c.currency = "USD"
            c.exchange = "SMART"  # use smart routing by default

        # Override attributes
        IBKRBroker.__update_ibkr_object(c, order.info.get("contract"))

        return c

    def _get_order(self, order: Order) -> IBKROrder:
        """Map an order to a IBKR order."""
        o = IBKROrder()
        o.action = "BUY" if order.is_buy else "SELL"
        o.totalQuantity = abs(order.size)
        if order.gtd:
            o.tif = "GTD"
            o.goodTillDate = order.gtd.strftime("%Y%m%d %H:%M:%S %Z")
        else:
            o.tif = "GTC"

        if order.limit:
            o.orderType = "LMT"
            o.lmtPrice = order.limit
        else:
            o.orderType = "MKT"

        # Override attributes
        IBKRBroker.__update_ibkr_object(o, order.info.get("order"))

        return o
