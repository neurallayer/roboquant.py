from dataclasses import dataclass
from decimal import Decimal
import logging

from roboquant.account import Account, Position
from roboquant.brokers.broker import Broker, _update_positions
from roboquant.event import Event, Quote, PriceItem
from roboquant.order import Order, OrderStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _Trx:
    """transaction for an executed trade"""

    symbol: str
    size: Decimal
    price: float  # denoted in the currency of the symbol


class SimBroker(Broker):
    """Implementation of a Broker that simulates order handling and trade execution.

    This class can be extended to support different types of use-cases, like margin trading.
    """

    def __init__(
        self,
        initial_deposit=1_000_000.0,
        price_type="OPEN",
        slippage=0.001,
        clean_up_orders=True,
    ):
        super().__init__()
        self._account = Account()
        self._modify_orders: list[Order] = []
        self._create_orders: dict[str, Order] = {}
        self._account.cash = initial_deposit
        self._account.buying_power = initial_deposit
        self._order_id = 0

        self.slippage = slippage
        self.price_type = price_type
        self.clean_up_orders = clean_up_orders
        self.initial_deposit = initial_deposit

    def reset(self):
        self._account = Account()
        self._modify_orders: list[Order] = []
        self._create_orders: dict[str, Order] = {}
        self._account.cash = self.initial_deposit
        self._account.buying_power = self.initial_deposit
        self._order_id = 0

    def _update_account(self, trx: _Trx):
        """Update a position and cash based on a new transaction"""
        acc = self._account
        symbol = trx.symbol
        acc.cash -= acc.contract_value(symbol, trx.price, trx.size)

        size = acc.get_position_size(symbol)

        if size.is_zero():
            # opening of position
            acc.positions[symbol] = Position(trx.size, trx.price, trx.price)
        else:
            new_size: Decimal = size + trx.size
            if new_size.is_zero():
                # closing of position
                del acc.positions[symbol]
            elif new_size.is_signed() != size.is_signed():
                # reverse of position
                acc.positions[symbol] = Position(new_size, trx.price, trx.price)
            else:
                # increase of position size
                old_price = acc.positions[symbol].avg_price
                avg_price = (old_price * float(size) + trx.price * float(trx.size)) / (float(size + trx.size))
                acc.positions[symbol] = Position(new_size, avg_price, trx.price)

    def _get_execution_price(self, order: Order, item: PriceItem) -> float:
        """Return the execution price to use for an order based on the price item.

        The default implementation is a fixed slippage percentage based on the configured price_type.
        """
        if isinstance(item, Quote):
            return item.ask_price if order.is_buy else item.bid_price

        price = item.price(self.price_type)
        correction = self.slippage if order.is_buy else -self.slippage
        return price * (1.0 + correction)

    def _execute(self, order: Order, item) -> _Trx | None:
        """Simulate a market execution for the three order types"""

        price = self._get_execution_price(order, item)
        fill = self._get_fill(order, price)
        if fill:
            return _Trx(order.symbol, fill, price)
        return None

    def __next_order_id(self):
        result = str(self._order_id)
        self._order_id += 1
        return result

    def _has_expired(self, order: Order) -> bool:
        """Returns true if the order has expired, false otherwise"""
        if not order.gtd:
            return False
        return self._account.last_update >= order.gtd

    def _get_fill(self, order: Order, price: float) -> Decimal:
        """Return the fill for the order given the provided price.

        The default implementation is:

        - A market order is always fully filled,
        - A limit order only when the limit is below the BUY price or above the SELL price.

        Overwrite this method in a subclass if you require more advanced behavior, like partial fills.
        """
        if order.is_buy and price <= order.limit:
            return order.remaining
        if order.is_sell and price >= order.limit:
            return order.remaining

        return Decimal(0)

    def place_orders(self, orders: list[Order]):
        """Place new orders at this broker. The order gets assigned a unique order-id if it hasn't one already.

        Orders that are placed that have already an order-id are either update- or cancellation-orders.

        There is no trading simulation yet performed or account updated. Orders placed at time `t`, will be
        processed during time `t+1`. This protects against future bias.
        """
        for order in orders:
            assert not order.is_closed, "cannot place a closed order"
            if order.id is None:
                order.id = self.__next_order_id()
                assert order.id not in self._create_orders
                self._create_orders[order.id] = order
            else:
                assert order.id in self._create_orders, "existing order id is not found"
                self._modify_orders.append(order)

    def _process_modify_order(self):
        for order in self._modify_orders:
            orig_order = self._create_orders.get(order.id)  # type: ignore
            if not orig_order:
                logger.info("couldn't find order with id %s", order.id)
                continue
            if orig_order.is_closed:
                logger.info("cannot modify order because order is already closed %s", orig_order)
                continue
            if order.is_cancellation:
                orig_order.status = OrderStatus.CANCELLED
            else:
                orig_order.size = order.size or orig_order.size
                orig_order.limit = order.limit or orig_order.limit
                logger.info("modified order %s", orig_order)

        self._modify_orders = []

    def _process_create_orders(self, prices: dict[str, PriceItem]):
        for order in self._create_orders.values():
            if order.is_closed:
                continue
            if self._has_expired(order):
                logger.info("order expired order=%s time=%s", order, self._account.last_update)
                order.status = OrderStatus.EXPIRED
            else:
                if (item := prices.get(order.symbol)) is not None:
                    trx = self._execute(order, item)
                    if trx is not None:
                        logger.info("executed order=%s trx=%s", order, trx)
                        self._update_account(trx)
                        order.fill += trx.size
                        if order.fill == order.size:
                            order.status = OrderStatus.FILLED

    def _calculate_open_orders(self):
        reserved = 0.0
        for order in self._account.open_orders:
            old_pos = self._account.get_position_size(order.symbol)
            remaining = order.size - order.fill

            # only update reserved amount if remaining order size would increase position size
            if abs(old_pos + remaining) > abs(old_pos):
                reserved += abs(self._account.contract_value(order.symbol, order.limit, remaining))

        return reserved

    def _calculate_short_positions(self):
        reserved = 0.0
        for symbol, position in self._account.short_positions().items():
            short_value = self._account.contract_value(symbol, position.mkt_price, position.size)
            reserved += abs(short_value)
        return reserved

    def _calculate_buyingpower(self):
        """Calculate buying power, based on:

        buying_power = cash - open_orders - short_positions
        """
        bp = self._account.cash
        bp -= self._calculate_open_orders()
        bp -= self._calculate_short_positions()
        return bp

    def sync(self, event: Event | None = None) -> Account:
        """This will perform the order-execution simulation for the open orders and
        return the updated the account as a result."""

        acc = self._account
        if event:
            acc.last_update = event.time

        prices = event.price_items if event else {}

        if self.clean_up_orders:
            # only keep the open orders from the previous step, remove the closed orders
            # this improves performance for large back tests with many orders
            self._create_orders = {order_id: order for order_id, order in self._create_orders.items() if order.is_open}

        self._process_modify_order()
        self._process_create_orders(prices)
        _update_positions(acc, event, self.price_type)

        acc.orders = list(self._create_orders.values())
        acc.buying_power = self._calculate_buyingpower()
        return acc

    def __repr__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"SimBroker({attrs})"
