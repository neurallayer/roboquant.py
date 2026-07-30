import logging
import ccxt
from array import array
from datetime import date, datetime, timezone
from typing import Any, override

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset, Crypto
from roboquant.common.event import Bar, Event
from roboquant.common.monetary import Amount, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position
from roboquant.feeds.in_memory_feed import InMemoryFeed

logger = logging.getLogger(__name__)


class CryptoFeed(InMemoryFeed):
    """retrieve historic crypto market data using the CCXT library. By default, it will retrieve daily data, but
    you can specify a different interval."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        *symbols: str,
        start_date: str | date | datetime = "2020-01-01T00:00:00",
        end_date: str | date | datetime | None = None,
        interval: str = "1d",
    ):
        """
        Create a new CryptoFeed instance
        Args:
            symbols: list of symbols to retrieve
            start_date: the start date of the data to retrieve, default in `2020-01-01`
            end_date: the end date of the data to retrieve, default is `None` (today)
            interval: the interval of the data to retrieve, default is `1d` (daily)
        """

        super().__init__()

        if not exchange.has["fetchOHLCV"]:
            raise ValueError(f"Exchange {exchange} does not support fetching OHLCV data")

        start_date = str(start_date)
        end_date = datetime.fromisoformat(str(end_date)).astimezone(timezone.utc) if end_date else None

        for symbol in symbols:
            try:
                asset = self._get_asset(symbol)
                logger.debug("requesting symbol=%s", symbol)
                done = False
                since = exchange.parse8601(start_date)

                while not done:
                    # fetch_ohlcv returns a list of lists, each containing [timestamp, open, high, low, close, volume]
                    rows: list[list[Any]] = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=interval,
                        since=since,
                        limit=None,
                    )

                    if not rows:
                        break

                    for row in rows:
                        dt = datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc)
                        if end_date and dt > end_date:
                            done = True
                            break
                        prices = row[1:6]
                        b = Bar(asset, array("f", prices), interval)
                        self._add_item(dt, b)

                    since = row[0] + 1

                    logger.info("retrieved symbol=%s items=%s last=%s", symbol, len(rows), dt)
            except Exception:
                logger.exception("Error retrieving symbol=%s", symbol, exc_info=True)

        self._update()

    def _get_asset(self, symbol: str) -> Asset:
        """Get the asset for the given symbol. The default implementation will return an
        asset of the type Crypto.
        Subclasses can override this method to provide a different asset type."""
        return Crypto.from_symbol(symbol)


class CryptoBroker(LiveBroker):
    """Broker that supports cryptocurrency exchanges using the ccxt library. Not all exchanges
    support all features, so check the documentation of the exchange you want to use. If a required feature is not supported,
    a `NotSupported` exception will be raised.
    """

    def __init__(self, exchange: ccxt.Exchange, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__client = exchange

    def connect(self):
        # Default implementation for connecting to the crypto exchange
        logger.info("Connecting to crypto exchange...")

    def disconnect(self):
        # Default implementation for disconnecting from the crypto exchange
        logger.info("Disconnecting from crypto exchange...")

    @override
    def _place_order(self, order: Order) -> None:
        # Default implementation for placing an order
        side = 'buy' if order.is_buy else 'sell'
        result = self.__client.create_order(
            symbol = order.asset.symbol,
            type =  'limit',
            side = side,
            amount = float(abs(order.size)),
            price = order.limit,
        )
        logger.info("result place order order=%s result=%s", order, result)

    @override
    def _get_account(self, event: Event | None = None) -> Account:
        """Sync the account object from the real broker. It requires that following
        methods are supported by your broker:
        - fetch_balance
        - fetch_open_orders
        - fetch_positions
        """

        account = Account()
        account.orders = self._get_open_orders()
        account.portfolio = self._get_positions()
        account.cash = self._get_balance()
        account.buying_power = self._get_buying_power()
        return account

    @override
    def _cancel_order(self, order: Order):
        # Default implementation for cancelling a
        order_id = order.id
        result = self.__client.cancel_order(order_id)
        logger.info("Cancelled order order_id=%s result=%s", order_id, result)
        return result

    def _get_balance(self) -> Wallet:
        # Default implementation for retrieving account balance
        result = self.__client.fetch_balance()
        w = Wallet()
        for currency, balance in result['free'].items():
            if balance > 0:
                w += Amount(currency, balance)
        return w

    def _get_buying_power(self) -> Amount:
        # Default implementation for retrieving account balance
        info = self.__client.fetch_balance()["info"]  # type: ignore
        return Amount(info["currency"], float(info["buying_power"]))

    def _get_open_orders(self) -> list[Order]:
        # Default implementation for retrieving open orders
        orders = self.__client.fetch_open_orders()
        result = []
        for order in orders:
            asset = Asset(order['symbol'])
            size = order['amount']
            limit = order['price']
            size = size if order['side'] == 'buy' else -size
            id = order["id"]
            o = Order(asset, size, limit, id = id)
            result.append(o)
        return result

    def _get_positions(self) -> Portfolio:
        result = Portfolio()
        try:
            positions = self.__client.fetch_positions()
        except ccxt.NotSupported as e:
            logger.error(e)
            return result

        for position in positions:
            size = position['amount']
            asset = Crypto.from_symbol(position['symbol'])
            size = position['amount']
            avg_entry_price = position['entry_price']
            p = Position(size, avg_entry_price, float("nan"))
            result[asset] = p
        return result

    @override
    def _update_order(self, order: Order) -> None:
        raise NotImplementedError
