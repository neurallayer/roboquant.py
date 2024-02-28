from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from roboquant.account import Account, Position
from roboquant.brokers.broker import Broker
from roboquant.event import Event
from roboquant.order import Order, OrderStatus


@dataclass(slots=True, frozen=True)
class _Trx:
    """transaction for an executed trade"""

    symbol: str
    size: Decimal
    price: float  # denoted in the currency of the symbol


@dataclass
class _OrderState:
    order: Order
    accepted: datetime | None = None


class SimBroker(Broker):
    """Implementation of a Broker that simulates order handling and trade execution.

    This class can be extended to support different types of use-cases, like margin trading.
    """

    def __init__(
            self, initial_deposit=1_000_000.0, account=None, price_type="DEFAULT", slippage=0.001, clean_up_orders=True
    ):
        super().__init__()
        self.initial_deposit = initial_deposit
        self._account = account or Account()
        self._modify_orders: list[Order] = []
        self._account.buying_power = initial_deposit
        self.slippage = slippage
        self.price_type = price_type
        self._prices: dict[str, float] = {}
        self._orders: dict[str, _OrderState] = {}
        self.clean_up_orders = clean_up_orders
        self.__order_id = 0

    def _update_account(self, trx: _Trx):
        """Update a position and cash based on a new transaction"""
        acc = self._account
        symbol = trx.symbol
        acc.buying_power -= acc.contract_value(symbol, trx.size, trx.price)

        size = acc.get_position_size(symbol)

        if size.is_zero():
            # opening of position
            acc.positions[symbol] = Position(trx.size, trx.price)
        else:
            new_size: Decimal = size + trx.size
            if new_size.is_zero():
                # closing of position
                del acc.positions[symbol]
            elif new_size.is_signed() != size.is_signed():
                # reverse of position
                acc.positions[symbol] = Position(new_size, trx.price)
            else:
                # increase of position size
                old_price = acc.positions[symbol].avg_price
                avg_price = (old_price * float(size) + trx.price * float(trx.size)) / (float(size + trx.size))
                acc.positions[symbol] = Position(new_size, avg_price)

    def _get_execution_price(self, order, item) -> float:
        """Return the execution price to use for an order based on the price item.

        The default implementation is a fixed slippage percentage based on the configured price_type.
        """

        price = item.price(self.price_type)
        correction = self.slippage if order.is_buy else -self.slippage
        return price * (1.0 + correction)

    def _simulate_market(self, order: Order, item) -> _Trx | None:
        """Simulate a market for the three order types"""

        price = self._get_execution_price(order, item)
        if self._is_executable(order, price):
            return _Trx(order.symbol, order.size, price)

    def __next_order_id(self):
        result = str(self.__order_id)
        self.__order_id += 1
        return result

    def _has_expired(self, state: _OrderState) -> bool:
        if state.accepted is None:
            return False
        else:
            return self._account.last_update - state.accepted > timedelta(days=180)

    def _is_executable(self, order, price) -> bool:
        """Is this order executable given the provided execution price.
        A market order is always executable, a limit order only when the limit is below the BUY price or
        above the SELL price"""
        if order.limit is None:
            return True
        if order.is_buy and price <= order.limit:
            return True
        if order.is_sell and price >= order.limit:
            return True

        return False

    def __update_mkt_prices(self, price_items):
        """track the latest market prices for all open positions"""
        for symbol in self._account.positions.keys():
            if item := price_items.get(symbol):
                self._prices[symbol] = item.price(self.price_type)

    def place_orders(self, orders):
        """Place new orders at this broker. The order gets assigned a unique id if it hasn't one already.

        There is no trading simulation yet performed or account updated. Orders placed at time `t`, will be
        processed during time `t+1`. This protects against future bias.
        """
        for order in orders:
            assert not order.closed, "cannot place closed orders"
            if order.id is None:
                order.id = self.__next_order_id()
                assert order.id not in self._orders
                self._orders[order.id] = _OrderState(order)
            else:
                assert order.id in self._orders, "existing order id not found"
                self._modify_orders.append(order)

    def _process_modify_order(self):
        for order in self._modify_orders:
            state = self._orders[order.id]  # type: ignore
            if state.order.closed:
                continue
            elif order.is_cancellation:
                state.order.status = OrderStatus.CANCELLED
            else:
                state.order.size = order.size or state.order.size
                state.order.limit = order.limit or state.order.limit
        self._modify_orders = []

    def _process_create_orders(self, prices):
        for state in self._orders.values():
            order = state.order
            if order.closed:
                continue
            if self._has_expired(state):
                order.status = OrderStatus.EXPIRED
            else:
                if (item := prices.get(order.symbol)) is not None:
                    state.accepted = state.accepted or self._account.last_update
                    trx = self._simulate_market(order, item)
                    if trx is not None:
                        self._update_account(trx)
                        order.status = OrderStatus.FILLED
                        order.fill = order.size

    def sync(self, event: Event | None = None) -> Account:
        """This will perform the trading simulation for open orders and update the account accordingly."""

        acc = self._account
        if event:
            acc.last_update = event.time

        prices = event.price_items if event else {}

        if self.clean_up_orders:
            # remove all the closed orders from the previous step
            self._orders = {order_id: state for order_id, state in self._orders.items() if not state.order.closed}

        self._process_modify_order()
        self._process_create_orders(prices)
        self.__update_mkt_prices(prices)

        acc.equity = acc.mkt_value(self._prices) + acc.buying_power
        acc.orders = [state.order for state in self._orders.values()]
        return acc
