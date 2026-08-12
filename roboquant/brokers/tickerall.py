import logging
import math
from decimal import Decimal
from typing import Self

from tickerall import Tickerall
from tickerall.types import BrokerName, TerminalType

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.monetary import Amount, Currency, USD, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position
from roboquant.feeds.tickerall import _require_tickerall_account_id, _to_asset

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

    There are two ways to construct the broker:

    - `TickerAllBroker.connect(...)` (recommended) takes MetaTrader credentials (`broker`, `server`,
      `account`, `password`), opens the broker session for you, and binds the resulting TickerAll
      `account_id`. It keeps the session alive (transparently re-arming it if the account goes cold), and
      `close` ends that session. This is the one-step path: credentials in, a working broker out — no
      separate `client.sessions.start(...)` call needed.
    - `TickerAllBroker(api_key, account_id)` takes an account you already connected yourself (via
      `client.sessions.start(...)` or the TickerAll dashboard). `close` leaves that session running — it
      only ends sessions that `connect` started.

    Build feeds cheaply from the broker's session by reusing `broker.account_id`, so the account's
    session is opened once rather than once per component. For example:

        broker = TickerAllBroker.connect(api_key, broker="mt5", server="Exness-MT5Trial", account=12345, password="pw")
        live = TickerAllLiveFeed(api_key, broker.account_id)
        hist = TickerAllHistoricFeed(api_key, broker.account_id)
        ...
        broker.close()  # ends the session connect() opened

    Args:
        api_key: the TickerAll api key.
        account_id: the id of an already-connected TickerAll broker account (see `connect` to open one
            from MetaTrader credentials instead).
        base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api.tickerall.com") -> None:
        super().__init__()
        _require_tickerall_account_id(account_id)
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)
        # True only when this broker opened the session itself (via `connect`); a session passed in by
        # account_id belongs to the caller and is never ended on `close`.
        self._owns_session = False

    @classmethod
    def connect(
        cls,
        api_key: str,
        *,
        broker: BrokerName,
        server: str,
        account: int | str,
        password: str,
        terminal_type: TerminalType | None = None,
        base_url: str = "https://api.tickerall.com",
    ) -> Self:
        """Connect a MetaTrader account by its credentials and return a broker bound to it.

        This does the session-start step for you, so you go straight from MetaTrader credentials to a
        working broker with no separate `client.sessions.start(...)` call. It uses the SDK's
        `sessions.keep_alive`, which opens the connection and transparently re-arms it if the account
        later goes cold (e.g. after a server restart) — the right choice for a long-running trading
        session. Pair it with `close`, which ends the session this method opened.

        Args:
            api_key: the TickerAll api key.
            broker: the MetaTrader platform, `"mt5"` or `"mt4"`.
            server: the broker server name (e.g. `"Exness-MT5Trial"`).
            account: the MetaTrader account login.
            password: the account password (used only to connect; never persisted).
            terminal_type: which client to present as — `"MOBILE"` (default) or `"CLIENT"`. (`"WEB"`
                also needs the broker's web-terminal URL, so open a WEB session with
                `client.sessions.start` directly.)
            base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
        """
        instance = cls(api_key, "", base_url)
        try:
            result = instance._client.sessions.keep_alive(
                broker=broker, server=server, account=account, password=password, terminal_type=terminal_type,
            )
        except Exception:
            instance._client.close()
            raise
        instance._account_id = result.account_id
        instance._owns_session = True
        return instance

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
        """Close the broker. If it was created with `connect`, this also ends the broker session it
        opened; a session you passed in by `account_id` is left running. Always closes the underlying SDK
        client (its HTTP connection pool)."""
        if self._owns_session and self._account_id:
            try:
                self._client.sessions.end(self._account_id)
            except Exception:
                logger.warning("failed to end TickerAll session %s on close", self._account_id, exc_info=True)
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
            # size = abs(float(o.volume))
            size = abs(Decimal(str(o.volume)))
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
