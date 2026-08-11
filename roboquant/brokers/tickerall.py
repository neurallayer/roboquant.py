import logging
import math
from decimal import Decimal

from tickerall import Tickerall

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.monetary import Amount, Currency, USD, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position
from roboquant.feeds.tickerall import _to_asset

logger = logging.getLogger(__name__)


class TickerAllBroker(LiveBroker):
    """Broker for TickerAll (https://tickerall.com), a hosted MetaTrader 4 and 5 API.

    This lets roboquant trade a MetaTrader broker account through TickerAll, without a local MetaTrader
    terminal and from any operating system. It is built on the official `tickerall` Python SDK. The account
    you connect at TickerAll is the source of truth; the broker syncs its cash, buying power and open
    positions, and places/updates/cancels orders against it.

    Following roboquant's order model, an order without an `id` is a new order, an order with an `id` and a
    non-zero size updates that resting pending order, and an order with an `id` and a zero size cancels it.
    A new order with a finite `limit` is placed as a pending limit order; a new order whose `limit` is
    `NaN` (`float("nan")`) is placed as a market order.

    Note that roboquant models one net position per asset, which maps to a MetaTrader **netting** account.
    Stop-loss / take-profit are not expressible in roboquant's order model and are out of scope.

    Args:
        api_key: the TickerAll api key.
        account_id: the id of the connected TickerAll broker account.
        base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api.tickerall.com") -> None:
        super().__init__()
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)

    @property
    def client(self) -> Tickerall:
        """The underlying `tickerall` SDK client, handy for account management not covered by the `Broker`
        interface (for example `broker.client.positions.close(account_id, ticket)`)."""
        return self._client

    @property
    def account_id(self) -> str:
        """The id of the connected TickerAll broker account this broker trades."""
        return self._account_id

    def close(self) -> None:
        """Close the underlying SDK client (its HTTP connection pool)."""
        self._client.close()

    def _get_account(self) -> Account:
        account = Account()
        detail = self._client.accounts.get(self._account_id)

        # The financials block (`account`) is present only when the account is connected/warm. A balance of
        # 0.0 is a valid (unfunded) account state, so warmth is keyed on the block being present, not on the
        # balance being non-zero.
        financials = detail.account
        if financials is None:
            raise ValueError(f"account {self._account_id} is not connected/warm; reconnect it first")

        currency = Currency(financials.currency or USD)
        free_margin = financials.free_margin
        balance = financials.balance
        account.buying_power = Amount(currency, float(free_margin if free_margin is not None else balance))
        account.cash = Wallet(Amount(currency, float(balance)))
        account.portfolio = self._sync_positions(detail.positions, currency)
        account.orders = self._sync_orders(currency)
        return account

    def _sync_positions(self, positions, currency: Currency) -> Portfolio:
        portfolio = Portfolio()
        for p in positions:
            if not p.symbol or p.volume is None:
                continue
            size = Decimal(str(p.volume))
            if str(p.side).upper() == "SELL":
                size = -size
            entry = float(p.entry_price or 0.0)
            mkt = float(p.current_price) if p.current_price is not None else entry
            portfolio[_to_asset(p.symbol, currency)] = Position(size, entry, mkt)
        return portfolio

    def _sync_orders(self, currency: Currency) -> list[Order]:
        orders: list[Order] = []
        for o in self._client.orders.list_pending(self._account_id):
            if not o.symbol or o.volume is None or o.ticket is None:
                continue
            limit = o.limit_price if o.limit_price is not None else o.price
            if limit is None:
                limit = float("nan")
            asset = _to_asset(o.symbol, currency)
            size = abs(float(o.volume))
            if str(o.side).upper() == "SELL":
                orders.append(self._sell_order(o.ticket, asset, size, limit, 0))
            else:
                orders.append(self._buy_order(o.ticket, asset, size, limit, 0))
        return orders

    def _place_order(self, order: Order) -> None:
        is_market = math.isnan(order.limit)
        result = self._client.orders.place(
            self._account_id,
            type="market" if is_market else "limit",
            symbol=order.asset.symbol,
            side="BUY" if order.is_buy else "SELL",
            volume=abs(float(order.size)),
            price=None if is_market else order.limit,
        )
        logger.info("placed order symbol=%s ticket=%s", order.asset.symbol, result.ticket)

    def _update_order(self, order: Order) -> None:
        self._client.orders.modify_pending(self._account_id, int(order.id), price=order.limit)

    def _cancel_order(self, order: Order) -> None:
        self._client.orders.cancel_pending(self._account_id, int(order.id))
