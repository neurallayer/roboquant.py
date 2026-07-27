from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import replace
import logging
from typing import override

from roboquant.portfolio import Position
from roboquant.account import Account, Trade
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
        fee: Amount = Amount(USD , 0.0)
    ):
        """Create a new SimBroker instance.

        Arguments:
            deposit: The initial deposit of cash in the account. The currency of the deposit is also as the base currency
                for the account. The default is 1,000,000 USD. If there are assets denoted in a different
                currency, a currency converter needs to be configured.
            price_type: The price type to use for the execution, like OPEN, CLOSE, HIGH or LOW. Default is OPEN.
            slippage: The slippage to use for the execution, a percentage value. Default is 0% (0.0)
            timezone: The timezone to use for to determine order expiration when the time in force for
                an order is set to `DAY`. Default is UTC.
            fee: Any additional fee or commmission that applies to each order execution.
        """

        super().__init__()
        self.slippage = slippage
        self.price_type = price_type
        self.deposit = deposit
        self.timezone = timezone
        self.fee = fee
        self.reset()

    def reset(self) -> None:
        """Reset the broker with the cash and buying power set to the initial deposit."""
        self._account = Account(self.deposit.currency)
        self._orders: dict[str, Order] = {}
        self._account.cash = Wallet(self.deposit)
        self._account.buying_power = self.deposit
        self._order_id = 0

    def _fee(self, asset: Asset, fill: Decimal, price: float, time: datetime) -> float:
        """Calculate any additional fee or commission, the default is zero.

        This is additional to any configured slippage. The slippage
        changes the execution price of the order while the fee only
        affects the cash balance and pnl.

        The returned fee should be denoted in the same currency as the asset.
        """
        return self.fee.convert_to(asset.currency, time)

    def __update_position(self, asset: Asset, fill: Decimal, price: float) -> float:
        """update position based on a fill and return the realized pnl"""
        assert fill != 0, "fill cannot be zero"

        pos = self._account.portfolio.get(asset, Position())

        new_size = pos.size + fill

        # position increase
        if (pos.size >= 0 and fill > 0) or (pos.size <= 0 and fill < 0):
            avg_price = (pos.avg_price * float(pos.size) + price * float(fill)) / float(new_size)
            pnl = 0.0

        # position decrease
        elif (pos.size >= 0 and new_size > 0) or (pos.size <= 0 and new_size < 0):
            avg_price = pos.avg_price
            pnl = asset.value(fill, price - pos.avg_price)

        # close or switch position side
        else:
            avg_price = price
            pnl = asset.value(pos.size, price - pos.avg_price)

        if new_size:
            self._account.portfolio[asset] = Position(new_size, avg_price, pos.mkt_price)
        else:
            del self._account.portfolio[asset]

        return pnl

    def __process_fill(self, asset: Asset, fill: Decimal, price: float, time: datetime) -> None:
        """Update the account positions, trades and cash based on a new fill"""
        if not fill:
            return

        acc = self._account
        acc.cash -= asset.amount(fill, price)
        fee = self._fee(asset, fill, price, time)
        acc.cash -= Amount(asset.currency, fee)
        pnl = self.__update_position(asset, fill, price) - fee
        trade = Trade(asset, time, fill, price, pnl)
        acc.trades.append(trade)

    def _get_execution_price(self, order: Order, item: PriceItem) -> float:
        """Return the execution price to use for an order based on the price item.

        The default implementation is a fixed slippage percentage based on the configured price_type.
        But is the price-item is a `Quote`, it will use the ask- en bid-price instead without
        any additional slippage.
        """
        if isinstance(item, Quote):
            return item.ask_price if order.is_buy else item.bid_price

        price = item.price(self.price_type)
        correction = self.slippage if order.is_buy else -self.slippage
        return price * (1.0 + correction)


    def _get_fill(self, order: Order, price: float) -> Decimal:
        """Simulate a market execution and return the filled size.
        This default implementation always fill the complete order.

        Overwrite this method in a subclass if a different simulation
        is required.
        """
        return order.remaining

    def __update_account_positions(self, event: Event) -> None:
        """Update the account with the latest market prices found in the event"""

        for asset, pos in self._account.portfolio.items():
            if price := event.get_price(asset, self.price_type):
                updated_position = Position(pos.size, pos.avg_price, price)
                self._account.portfolio[asset] = updated_position

    def __next_order_id(self) -> str:
        """Generate a new order id. The order id is a simple integer that is incremented for each new order."""
        result = str(self._order_id)
        self._order_id += 1
        return result

    @override
    def place_orders(self, orders: list[Order]) -> None:
        """Place new orders at this broker. The orders get assigned a unique order-id if there isn't one already.

        Orders that are placed that have already an order-id are either update- or cancellation-orders.

        There is no trading simulation yet performed or an account updated. This is done during the `sync` method.
        Orders placed at time `t`, will be processed during time `t+1`. This protects against future bias.
        """
        for order in orders:
            if not order.id:
                assert order.size != 0, "order size of a new order cannot be zero"
                order = replace(order, id=self.__next_order_id())

            self._orders[order.id] = order


    def _order_expired(self, order: Order, time: datetime) -> bool:
        """Check if the order is expired given the provided time"""

        # GTC order never expire
        if order.tif == "GTC" or not order.time:
            return False

        # We are now in the DAY tif branch
        order_date = order.time.astimezone(self.timezone).date()
        exchange_date = time.astimezone(self.timezone).date()

        return exchange_date > order_date

    def __process_orders(self, event: Event) -> None:
        if not self._orders:
            return

        prices = event.price_items
        time = event.time

        orders: dict[str, Order] = {}

        for order in self._orders.values():

            if self._order_expired(order, time):
                logger.info("expired order %s", order)
                continue

            if item := prices.get(order.asset):
                if not order.time:
                    order = replace(order, time = time)
                price = self._get_execution_price(order, item)
                if order.is_executable(price):
                    fill = self._get_fill(order, price)
                    logger.info("executed order=%s fill=%s", order, fill)
                    order = replace(order, fill = order.fill + fill)
                    self.__process_fill(order.asset, fill, price, time)

            if order.remaining:
                orders[order.id] = order


        self._orders = orders

    def _calculate_open_orders(self) -> Wallet:
        """Calculate the buying power required for the open orders"""
        result = Wallet()
        for order in self._account.orders:
            if order.is_buy and order.remaining:
                result += order.asset.amount(order.remaining, order.limit)
        return result

    def _calculate_short_positions(self) -> Wallet:
        reserved = Wallet()
        for asset, position in self._account.portfolio.short_positions().items():
            short_value = asset.amount(-position.size, position.mkt_price)
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

    @override
    def sync(self, event: Event | None = None) -> Account:
        """This will perform the order-execution simulation for the open orders and
        return the updated the account as a result."""

        if event:
            self._account.last_update = event.time
            self.__process_orders(event)
            self.__update_account_positions(event)

        acc = self._account
        acc.orders = list(self._orders.values())
        acc.buying_power = self._calculate_buyingpower()
        return acc

    def __repr__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"SimBroker({attrs})"
