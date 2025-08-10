import logging
import ccxt
import os
from roboquant.account import Account, Position
from roboquant.asset import Asset
from roboquant.brokers.broker import LiveBroker, Order
from roboquant.event import Event
from roboquant.monetary import Wallet, Amount, USD

from dotenv import load_dotenv

load_dotenv()


def _get_credentials():
    return os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET"]

logger = logging.getLogger(__name__)


class CryptoBroker(LiveBroker):
    """Broker that supports cryptocurrency exchanges using the ccxt library.
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
        account = Account()
        account.orders = self._get_open_orders()
        account.positions = self._get_positions()
        account.cash = self._get_balance()
        account.buying_power = self._get_buying_power()
        return account

    def _cancel_order(self, order: Order):
        # Default implementation for cancelling a
        order_id = order.id
        try:
            result = self.__client.cancel_order(order_id)  # type: ignore
            logger.info("Cancelled order order_id=%s result=%s", order_id, result)
            return result
        except Exception as e:
            logger.error("Failed to cancel order order_id=%s error=%s", order_id, e)
            raise

    def _get_balance(self) -> Wallet:
        # Default implementation for retrieving account balance
        result = self.__client.fetch_balance()
        logger.info("Fetched balance: %s", result)
        return Wallet()

    def _get_buying_power(self) -> Amount:
        # Default implementation for retrieving account balance
        return 1000.0@USD

    def _get_open_orders(self) -> list[Order]:
        # Default implementation for retrieving open orders
        result = self.__client.fetch_open_orders()
        return result

    def _get_positions(self) -> dict[Asset, Position]:
        result = self.__client.fetch_positions()
        return result

    def _update_order(self, order: Order):
        raise NotImplementedError


if __name__ == "__main__":
    # Set logging at higher level
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    key, secret = _get_credentials()
    exchange = ccxt.alpaca({
        "apiKey" : key,
        "secret": secret
    })
    exchange.set_sandbox_mode(True)
    # exchange = ccxt.kraken()  # or any other exchange supported by ccxt
    broker = CryptoBroker(exchange)
    acc = broker.sync()
    print(acc)
