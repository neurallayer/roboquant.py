from datetime import date, datetime, timezone
from decimal import Decimal
import logging

from roboquant.account import Account, Position, Trade
from roboquant.asset import Asset
from roboquant.brokers.broker import Broker
from roboquant.event import Event, Quote, PriceItem
from roboquant.order import Order
from roboquant.monetary import Amount, Wallet, USD

logger = logging.getLogger(__name__)


class SimBroker(Broker):
    """Implementation of a Broker that simulates order execution and can be used in back tests.

    This class can be extended to support different types of use-cases, like margin trading.
    """

    def __init__(
        self,
        deposit: Amount = Amount(USD, 1_000_000.0),
        price_type: str = "OPEN",
        slippage: float = 0.0,
        timezone: timezone = timezone.utc,
    ):
        """Create a new SimBroker instance.
        params:
        - deposit: The initial deposit of cash in the account. The currency of the deposit is also as the base currency
        for the account. The default is 1,000,000 USD.
        - price_type: The price type to use for the execution, like OPEN, CLOSE, HIGH or LOW. Default is OPEN.
        - slippage: The slippage to use for the execution, a percentage value. Default is 0% (0.0)
        - timezone: The timezone to use for to determine order expiration when the order time in force is set to `DAY`.
            Default is UTC.
        """

        super().__init__()
        self._account = Account(deposit.currency)
        self._modify_orders: list[Order] = []
        self._create_orders: list[Order] = []
        self._account.cash = Wallet(deposit)
        self._account.buying_power = deposit
        self._order_id = 0

        self.slippage = slippage
        self.price_type = price_type
        self.deposit = deposit
        self._order_entry: dict[str, date] = {}
        self.timezone = timezone

    def reset(self):
        """Reset the broker with the cash and buying power set to the initial deposit."""
        self._account = Account(self.deposit.currency)
        self._modify_orders: list[Order] = []
        self._create_orders: list[Order] = []
        self._account.cash = Wallet(self.deposit)
        self._account.buying_power = self.deposit
        self._order_id = 0
        self._order_entry: dict[str, date] = {}

    def _fee(self, trade: Trade) -> Amount:
        """Calculate any additional transaction fee, default is zero"""
        return Amount(trade.asset.currency, 0.0)

    def _pnl(self, asset: Asset, size: Decimal, price: float) -> float:
        """
        Calculate the profit and loss for a given asset, size, and price. It takes into account the average price
        of the position if it exists, otherwise it returns 0.0
        Args:
            asset (Asset): The asset for which to calculate the profit and loss.
            size (Decimal): The size of the position.
            price (float): The current price of the asset.
        """
        pos = self._account.positions.get(asset)
        if not pos:
            return 0.0
        return asset.contract_value(size, price - pos.avg_price)


    def _update_account(self, trade: Trade):
        """Update a position, trades and cash based on a new trade"""
        acc = self._account
        acc.trades.append(trade)
        asset = trade.asset
        acc.cash -= asset.contract_amount(trade.size, trade.price)
        acc.cash -= self._fee(trade)

        size = acc.get_position_size(asset)

        if size.is_zero():
            # opening of position
            acc.positions[asset] = Position(trade.size, trade.price, trade.price)
        else:
            new_size: Decimal = size + trade.size
            if new_size.is_zero():
                # closing of position
                del acc.positions[asset]
            elif new_size.is_signed() != size.is_signed():
                # reverse of position
                acc.positions[asset] = Position(new_size, trade.price, trade.price)
            else:
                # increase of position size
                old_price = acc.positions[asset].avg_price
                avg_price = (old_price * float(size) + trade.price * float(trade.size)) / (float(size + trade.size))
                acc.positions[asset] = Position(new_size, avg_price, trade.price)

    def _get_execution_price(self, order: Order, item: PriceItem) -> float:
        """Return the execution price to use for an order based on the price item.

        The default implementation is a fixed slippage percentage based on the configured price_type.
        """
        if isinstance(item, Quote):
            return item.ask_price if order.is_buy else item.bid_price

        price = item.price(self.price_type)
        correction = self.slippage if order.is_buy else -self.slippage
        return price * (1.0 + correction)

    def _execute(self, order: Order, item: PriceItem, time: datetime) -> Trade | None:
        """Simulate a market execution and return a Trade object if the order has (partially) been executed."""

        price = self._get_execution_price(order, item)
        executable = price <= order.limit if order.is_buy else price >= order.limit
        if executable:
            pnl = self._pnl(order.asset, order.remaining, price)
            return Trade(order.asset, time, order.remaining, price, pnl)
        return None

    @staticmethod
    def _update_positions(account: Account, event: Event | None, price_type: str = "DEFAULT"):
        """Update the open positions in the account with the latest market prices found in the event"""
        if not event:
            return

        account.last_update = event.time

        for asset, position in account.positions.items():
            if price := event.get_price(asset, price_type):
                position.mkt_price = price

    def __next_order_id(self):
        """Generate a new order id. The order id is a simple integer that is incremented for each new order."""
        result = str(self._order_id)
        self._order_id += 1
        return result

    def place_orders(self, orders: list[Order]):
        """Place new orders at this broker. The orders get assigned a unique order-id if there isn't one already.

        Orders that are placed that have already an order-id are either update- or cancellation-orders.

        There is no trading simulation yet performed or an account updated. This is done during the `sync` method.
        Orders placed at time `t`, will be processed during time `t+1`. This protects against future bias.
        """
        for order in orders:
            if not order.id:
                order.id = self.__next_order_id()
                assert order.size != 0, "order size of a new order cannot be zero"
                self._create_orders.append(order)
            else:
                self._modify_orders.append(order)

    def _remove_order(self, order: Order):
        """Remove an order from the account, called when an order is completed, expired or cancelled."""
        self._account.orders.remove(order)
        self._order_entry.pop(order.id)

    def _process_modify_orders(self):
        """Process the modify orders. Modify orders will allways be handled and not propagate to the next sync call."""

        for order in self._modify_orders:
            orig_order = next((o for o in self._account.orders if o.id == order.id), None)
            if not orig_order:
                logger.info("couldn't find order with id %s", order.id)
                continue
            if order.size.is_zero():
                logger.info("cancelled order %s", orig_order)
                self._remove_order(orig_order)
            else:
                # update the order
                orig_order.size = order.size or orig_order.size
                orig_order.limit = order.limit or orig_order.limit
                logger.info("modified order %s", orig_order)

        self._modify_orders = []

    def _order_is_expired(self, order: Order, time: datetime) -> bool:
        if order.tif == "GTC":
            return False

        # We are now in the DAY tif branch
        if entry_time := self._order_entry.get(order.id):
            return time.astimezone(self.timezone).date() > entry_time
        else:
            # The first time we see this order
            self._order_entry[order.id] = time.astimezone(self.timezone).date()

        return False

    def _process_open_orders(self, event: Event | None):
        if not event or not self._account.orders:
            return
        prices = event.price_items

        for order in self._account.orders.copy():
            if self._order_is_expired(order, event.time):
                logger.info("expired order %s", order)
                self._remove_order(order)
            elif item := prices.get(order.asset):
                trade = self._execute(order, item, event.time)
                if trade is not None:
                    logger.info("executed order=%s trx=%s", order, trade)
                    self._update_account(trade)
                    order.fill += trade.size
                    if not order.remaining:
                        logger.info("completed order %s", order)
                        self._remove_order(order)

    def _calculate_open_orders(self):
        """Calculate the buying power required for the open orders"""
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
        """Calculate the buying power.
        The default implementation uses the following calculation:

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
