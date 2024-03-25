import logging
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import TradeAccount
from alpaca.trading.models import Position as APosition
from alpaca.trading.models import Order as AOrder
from alpaca.trading.models import OrderStatus as AOrderStatus

from alpaca.trading.requests import MarketOrderRequest, ReplaceOrderRequest
from roboquant.account import Account, Position
from roboquant.config import Config
from roboquant.event import Event
from roboquant.brokers.broker import Broker
from roboquant.order import Order, OrderStatus


logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):

    def __init__(self, api_key=None, secret_key=None) -> None:
        self.__account = Account()
        config = Config()
        api_key = api_key or config.get("alpaca.public.key")
        secret_key = secret_key or config.get("alpaca.secret.key")
        self.__client = TradingClient(api_key, secret_key)
        self.__has_new_orders_since_sync = False
        self.price_type = "DEFAULT"
        self.sleep_after_cancel = 0.0

    def _should_sync(self, now: datetime):
        """Avoid too many API calls"""
        return self.__has_new_orders_since_sync or now - self.__account.last_update > timedelta(seconds=1)

    def _sync_orders(self):
        for order in self.__account.open_orders():
            assert order.id is not None
            alpaca_order: AOrder = self.__client.get_order_by_id(order.id)  # type: ignore
            order.fill = Decimal(alpaca_order.filled_qty)  # type: ignore
            match alpaca_order.status:
                case AOrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                case AOrderStatus.REJECTED:
                    order.status = OrderStatus.REJECTED
                case AOrderStatus.EXPIRED:
                    order.status = OrderStatus.EXPIRED
                case _:
                    order.status = OrderStatus.ACTIVE

    def _sync_positions(self):
        open_pos: list[APosition] = self.__client.get_all_positions()  # type: ignore
        self.__account.positions.clear()
        for p in open_pos:
            size = Decimal(p.qty)
            if p.side == "short":
                size = -size
            new_pos = Position(size, float(p.avg_entry_price), float(p.current_price or "nan"))
            self.__account.positions[p.symbol] = new_pos

    def sync(self, event: Event | None = None) -> Account:

        logger.debug("start sync")
        now = datetime.now(timezone.utc)

        if event:
            # Let make sure we don't use IBKRBroker by mistake during a back-test.
            if now - event.time > timedelta(minutes=30):
                logger.critical("received event from the past, now=%s event-time=%s", now, event.time)
                raise ValueError(f"received event too far in the past now={now} event-time={event.time}")

        client = self.__client
        acc: TradeAccount = client.get_account()  # type: ignore
        self.__account.buying_power = float(acc.buying_power)  # type: ignore
        self.__account.cash = float(acc.cash)  # type: ignore
        self.__account.last_update = now

        self._sync_positions()
        self._sync_orders()
        logger.debug("end sync")
        return self.__account

    def place_orders(self, orders):

        self.__has_new_orders_since_sync = len(orders) > 0

        for idx, order in enumerate(orders, start=1):
            if idx % 25 == 0:
                # avoid to many API calls
                time.sleep(1)

            assert order.is_open, "can only place open orders"
            if order.size.is_zero():
                assert order.id is not None, "can only cancel orders with an id"
                self.__client.cancel_order_by_id(order.id)
                if self.sleep_after_cancel:
                    time.sleep(self.sleep_after_cancel)
            else:

                if order.id is None:
                    req = self.get_request(order)
                    alpaca_order = self.__client.submit_order(req)
                    order.id = alpaca_order.id  # type: ignore
                    self.__account.orders.append(order)
                else:
                    req = self.get_replace_req(order)
                    self.__client.replace_order_by_id(order.id, req)

    def get_request(self, order: Order):
        size = OrderSide.BUY if order.size > 0 else OrderSide.SELL
        result = MarketOrderRequest(symbol=order.symbol, qty=abs(float(order.size)), side=size, time_in_force=TimeInForce.GTC)
        return result

    def get_replace_req(self, order: Order):
        result = ReplaceOrderRequest(qty=int(abs(float(order.size))), limit_price=order.limit)
        return result


if __name__ == "__main__":
    broker = AlpacaBroker()
    account = broker.sync()
    print(account)
    tsla_order = Order("TSLA", 10)
    broker.place_orders([tsla_order])
    time.sleep(5)
    account = broker.sync()
    print(account)
