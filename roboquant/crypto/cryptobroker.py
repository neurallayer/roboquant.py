import logging
import ccxt
from roboquant.account import Account, Position
from roboquant.asset import Asset, Crypto
from roboquant.brokers.broker import LiveBroker, Order
from roboquant.event import Event
from roboquant.monetary import Wallet, Amount

from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


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

    def _place_order(self, order: Order):
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

    def _get_account(self, event: Event | None = None) -> Account:
        """Sync the account object from the real broker. It requires that following
        methods are supported by your broker:
        - fetch_balance
        - fetch_open_orders
        - fetch_positions
        """

        account = Account()
        account.orders = self._get_open_orders()
        account.positions = self._get_positions()
        account.cash = self._get_balance()
        account.buying_power = self._get_buying_power()
        return account

    def _cancel_order(self, order: Order):
        # Default implementation for cancelling a
        order_id = order.id
        result = self.__client.cancel_order(order_id)  # type: ignore
        logger.info("Cancelled order order_id=%s result=%s", order_id, result)
        return result

    def _get_balance(self) -> Wallet:
        # Default implementation for retrieving account balance
        result = self.__client.fetch_balance()  # type: ignore
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
        orders = self.__client.fetch_open_orders() # type: ignore
        result = []
        for order in orders:
            asset = Asset(order['symbol'])
            size = order['amount']
            limit = order['price']
            size = size if order['side'] == 'buy' else -size
            o = Order(asset, size, limit)
            o.id = order['id']
            result.append(o)
        return result

    def _get_positions(self) -> dict[Asset, Position]:
        result = {}
        try:
            positions = self.__client.fetch_positions() # type: ignore
        except ccxt.NotSupported as e:
            logger.error(e)
            return result

        for position in positions:
            size = position['amount']
            asset = Crypto.from_symbol(position['symbol'])
            size = position['amount']
            avg_entry_price = position['entry_price']
            p = Position(asset, size, avg_entry_price)
            result[asset] = p
        return result

    def _update_order(self, order: Order) -> None:
        raise NotImplementedError

