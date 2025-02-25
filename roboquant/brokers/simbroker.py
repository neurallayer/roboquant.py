from dataclasses import dataclass
from decimal import Decimal
import logging

from roboquant.account import Account, Position
from roboquant.asset import Asset
from roboquant.brokers.broker import Broker
from roboquant.event import Event, Quote, PriceItem
from roboquant.order import Order
from roboquant.monetary import Amount, Wallet, USD

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _Trx:
    """transaction for an executed trade"""

    asset: Asset
    """The asset that was traded"""

    size: Decimal
    """The size of the trade"""

    price: float
    """The price of the trade denoted in the currency of the asset"""

class SimBroker(Broker):
    """Implementation of a Broker that simulates order execution and can be used in back tests.

    This class can be extended to support different types of use-cases, like margin trading.
    """

    def __init__(self, initial_deposit: Amount=Amount(USD, 1_000_000.0), price_type:str="OPEN", slippage:float=0.001):
        """Create a new SimBroker instance.
        params:
        - initial_deposit: The initial deposit of cash in the account. The currency of the deposit is used as the base currency
        for the account.
        - price_type: The price type to use for the execution, like OPEN, CLOSE, HIGH, LOW
        - slippage: The slippage to use for the execution, a percentage value. Default is 0.1% (0.001)
        """


        super().__init__()
        self._account = Account(initial_deposit.currency)
        self._modify_orders: list[Order] = []
        self._create_orders: list[Order] = []
        self._account.cash = Wallet(initial_deposit)
        self._account.buying_power = initial_deposit
        self._order_id = 0

        self.slippage = slippage
        self.price_type = price_type
        self.initial_deposit = initial_deposit

    def reset(self):
        """Reset the broker with the cash and buying power set to the initial deposit."""
        self._account = Account(self.initial_deposit.currency)
        self._modify_orders: list[Order] = []
        self._create_orders: list[Order] = []
        self._account.cash = Wallet(self.initial_deposit)
        self._account.buying_power = self.initial_deposit
        self._order_id = 0

    def _update_account(self, trx: _Trx):
        """Update a position and cash based on a new transaction"""
        acc = self._account
        asset = trx.asset
        acc.cash -= asset.contract_amount(trx.size, trx.price)

        size = acc.get_position_size(asset)

        if size.is_zero():
            # opening of position
            acc.positions[asset] = Position(trx.size, trx.price, trx.price)
        else:
            new_size: Decimal = size + trx.size
            if new_size.is_zero():
                # closing of position
                del acc.positions[asset]
            elif new_size.is_signed() != size.is_signed():
                # reverse of position
                acc.positions[asset] = Position(new_size, trx.price, trx.price)
            else:
                # increase of position size
                old_price = acc.positions[asset].avg_price
                avg_price = (old_price * float(size) + trx.price * float(trx.size)) / (float(size + trx.size))
                acc.positions[asset] = Position(new_size, avg_price, trx.price)

    def _get_execution_price(self, order: Order, item: PriceItem) -> float:
        """Return the execution price to use for an order based on the price item.

        The default implementation is a fixed slippage percentage based on the configured price_type.
        """
        if isinstance(item, Quote):
            return item.ask_price if order.is_buy else item.bid_price

        price = item.price(self.price_type)
        correction = self.slippage if order.is_buy else -self.slippage
        return price * (1.0 + correction)

    def _execute(self, order: Order, item: PriceItem) -> _Trx | None:
        """Simulate a market execution for the three order types"""

        price = self._get_execution_price(order, item)
        fill = self._get_fill(order, price)
        if fill:
            return _Trx(order.asset, fill, price)
        return None

    def __next_order_id(self):
        result = str(self._order_id)
        self._order_id += 1
        return result

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
        """Place new orders at this broker. The orders get assigned a unique order-id if there isn't one already.

        Orders that are placed that have already an order-id are either update- or cancellation-orders.

        There is no trading simulation yet performed or account updated. This is done during the `sync` method.
        Orders placed at time `t`, will be processed during time `t+1`. This protects against future bias.
        """
        for order in orders:
            if order.id is None:
                order.id = self.__next_order_id()
                self._create_orders.append(order)
            else:
                self._modify_orders.append(order)

    def _remove_order(self, order: Order):
        """Remove an order from the account, called when an order is completed, expired or cancelled."""
        self._account.orders.remove(order)

    def _process_modify_orders(self):
        for order in self._modify_orders:
            orig_order = next((o for o in self._account.orders if o.id == order.id), None)
            if not orig_order:
                logger.info("couldn't find order with id %s", order.id)
                continue
            if order.is_cancellation:
                logger.info("cancelled order %s", orig_order)
                self._remove_order(orig_order)
            else:
                # update the order
                orig_order.size = order.size or orig_order.size
                orig_order.limit = order.limit or orig_order.limit
                logger.info("modified order %s", orig_order)

        self._modify_orders = []

    def _process_open_orders(self, event: Event | None):
        if not event or not self._account.orders:
            return
        prices = event.price_items

        for order in self._account.orders.copy():
            if order.is_expired(event.time):
                logger.info("expired order %s", order)
                self._remove_order(order)
            elif item := prices.get(order.asset):
                trx = self._execute(order, item)
                if trx is not None:
                    logger.info("executed order=%s trx=%s", order, trx)
                    self._update_account(trx)
                    order.fill += trx.size
                    if order.completed:
                        logger.info("completed order %s", order)
                        self._remove_order(order)

    def _calculate_open_orders(self):
        result = Wallet()
        for order in self._account.orders:
            result += self._account.required_buying_power(order)
        return result

    def _calculate_short_positions(self):
        reserved = Wallet()
        for asset, position in self._account.short_positions().items():
            short_value = asset.contract_amount(-position.size, position.mkt_price)
            reserved += short_value
        return reserved

    def _calculate_buyingpower(self) -> Amount:
        """Calculate buying power, based on:

        buying_power = cash - open_orders - short_positions
        """
        result = Wallet()
        result += self._account.cash
        result -= self._calculate_open_orders()
        result -= self._calculate_short_positions()
        return Amount(self._account.base_currency, self._account.convert(result))

    def sync(self, event: Event | None = None) -> Account:
        """This will perform the order-execution simulation for the open orders and
        return the updated the account as a result."""

        acc = self._account
        if event:
            acc.last_update = event.time

        acc.orders += self._create_orders
        self._create_orders = []

        self._process_modify_orders()
        self._process_open_orders(event)
        self._update_positions(acc, event, self.price_type)
        acc.buying_power = self._calculate_buyingpower()
        return acc

    def __repr__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"SimBroker({attrs})"
