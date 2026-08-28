import logging
from collections import defaultdict
from decimal import Decimal
from typing import Self

from tickerall import Tickerall
from tickerall.types import BrokerName, Position as TaPosition, SymbolSpec, TerminalType

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.monetary import Amount, Currency, USD, Wallet
from roboquant.common.asset import Asset
from roboquant.common.order import Order
from roboquant.common.position import Position
from roboquant.common.timeframe import utcnow
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
    A new order with a `limit` price is placed as a pending limit order; a new order whose `limit` is
    `None` (`is_mkt_order`) is placed as a market order.

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
        # Lazily-loaded cache of the broker's symbol specs (lot step/min), for volume quantization.
        self._symbol_specs_cache: dict[str, SymbolSpec] | None = None
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
        buying_power = Amount(currency, float(free_margin if free_margin is not None else balance))
        cash = Wallet(Amount(currency, float(balance)))
        positions = self._sync_positions(detail.positions, currency)
        orders = self._sync_orders(currency)
        return Account(
            buying_power=buying_power,
            cash=cash,
            positions=positions,
            orders=orders,
            trades=[],
            last_update=utcnow()
        )

    def _sync_positions(self, positions, currency: Currency) -> list[Position]:
        # A MetaTrader account may be HEDGING (one broker ticket per trade) rather than netting. roboquant
        # models one NET position per asset, so aggregate all of a symbol's tickets into a single net
        # Position — otherwise same-symbol tickets overwrite each other and roboquant sees only the last
        # ticket instead of the net exposure.
        groups: dict[Asset, list[TaPosition]] = defaultdict(list)
        for p in positions:
            if not p.symbol or p.volume is None:
                continue
            groups[_to_asset(p.symbol, self._quote_currency(p.symbol), currency)].append(p)

        portfolio = []
        for asset, tickets in groups.items():
            net = Decimal("0")
            for p in tickets:
                vol = Decimal(str(p.volume))
                net += vol if str(p.side).upper() == "BUY" else -vol
            if net == 0:
                continue  # fully hedged flat — no net exposure to report
            net_long = net > 0
            # Weighted-average entry over the net-side tickets (the side matching the net sign). Under
            # active net-emulation all tickets are one-sided, so this is the exact average entry.
            num = Decimal("0")
            den = Decimal("0")
            for p in tickets:
                if (str(p.side).upper() == "BUY") == net_long:
                    vol = Decimal(str(p.volume))
                    num += vol * Decimal(str(p.entry_price or 0.0))
                    den += vol
            entry = float(num / den) if den else 0.0
            mkt = float(tickets[0].current_price) if tickets[0].current_price is not None else entry
            portfolio.append(Position(asset, net, entry, mkt))
        return portfolio

    def _sync_orders(self, currency: Currency) -> list[Order]:
        orders: list[Order] = []
        for o in self._client.orders.list_pending(self._account_id):
            if not o.symbol or o.volume is None or o.ticket is None:
                continue
            limit = o.limit_price if o.limit_price is not None else o.price
            if limit is None:
                limit = float("nan")
            asset = _to_asset(o.symbol, self._quote_currency(o.symbol), currency)
            # Decimal via str(), not float, keeps the size exact (see _sync_positions).
            size = abs(Decimal(str(o.volume)))
            if str(o.side).upper() == "SELL":
                orders.append(self._sell_order(o.ticket, asset, size, limit, 0))
            else:
                orders.append(self._buy_order(o.ticket, asset, size, limit, 0))
        return orders

    def _place_order(self, order: Order) -> None:
        # roboquant expresses a close/reduce/reverse as a new opposite-signed order. On a NETTING account
        # the broker nets it; on a HEDGING account a raw opposite order opens a SECOND ticket instead of
        # closing. So when a MARKET order opposes the current net, emulate netting: close the net-side
        # tickets (FIFO) up to the order size, then open any remainder (a reversal past flat). A limit
        # opposer can't be a market close, so it's placed as a normal pending order and nets when it fills.
        tickets = self._open_tickets(order.asset.symbol)
        net = Decimal("0")
        for t in tickets:
            vol = Decimal(str(t.volume))
            net += vol if str(t.side).upper() == "BUY" else -vol
        opposing = (net > 0 and order.is_sell) or (net < 0 and order.is_buy)
        if not opposing or not order.is_mkt_order:
            return self._place_raw(order)

        remaining = abs(order.size)
        net_long = net > 0
        for t in sorted(tickets, key=lambda x: (x.open_time or "", x.ticket)):
            if remaining <= 0:
                break
            if (str(t.side).upper() == "BUY") != net_long:
                continue  # only close net-side exposure; leave any already-opposite tickets alone
            vol = Decimal(str(t.volume))
            take = self._quantize_volume(order.asset.symbol, min(vol, remaining))
            if take <= 0:
                continue
            if take >= vol:
                self._client.positions.close(self._account_id, t.ticket)
            else:
                self._client.positions.close(self._account_id, t.ticket, volume=float(take))
            remaining -= take

        remaining = self._quantize_volume(order.asset.symbol, remaining)
        if remaining > 0:
            # Reversal past flat: open a fresh position for the leftover in the order's direction.
            self._client.orders.place(
                self._account_id,
                type="market",
                symbol=order.asset.symbol,
                side="BUY" if order.is_buy else "SELL",
                volume=float(remaining),
                price=None,
            )
        logger.info("net-emulated order symbol=%s size=%s net_before=%s", order.asset.symbol, order.size, net)

    def _place_raw(self, order: Order) -> None:
        """Place an order straight through to the broker (the original, netting-account behavior)."""
        is_market = order.is_mkt_order
        result = self._client.orders.place(
            self._account_id,
            type="market" if is_market else "limit",
            symbol=order.asset.symbol,
            side="BUY" if order.is_buy else "SELL",
            volume=abs(float(order.size)),
            price=None if is_market else order.limit,
        )
        logger.info("placed order symbol=%s ticket=%s", order.asset.symbol, result.ticket)

    def _open_tickets(self, symbol: str) -> list[TaPosition]:
        """Fresh open-position tickets for one symbol, read live (not from the possibly-stale account cache)."""
        detail = self._client.accounts.get(self._account_id)
        return [p for p in detail.positions if p.symbol == symbol and p.volume is not None]

    def _quantize_volume(self, symbol: str, vol: Decimal) -> Decimal:
        """Round a volume DOWN to the symbol's lot step; return 0 if it falls below the minimum lot."""
        if vol <= 0:
            return Decimal("0")
        spec = self._symbol_spec(symbol)
        if spec is None:
            return vol
        step = Decimal(str(spec.volume_step)) if getattr(spec, "volume_step", None) else Decimal("0")
        vmin = Decimal(str(spec.volume_min)) if getattr(spec, "volume_min", None) else Decimal("0")
        q = (vol // step) * step if step > 0 else vol
        if vmin > 0 and q < vmin:
            return Decimal("0")
        return q

    def _symbol_spec(self, symbol: str):
        """Lazily fetch + cache the broker's symbol specs, keyed by symbol name."""
        if self._symbol_specs_cache is None:
            cache: dict[str, SymbolSpec] = {}
            try:
                for s in self._client.accounts.symbol_specs(self._account_id):
                    name = getattr(s, "name", None)
                    if name:
                        cache[name] = s
            except Exception:
                logger.warning("failed to load symbol specs for %s", self._account_id, exc_info=True)
            self._symbol_specs_cache = cache
        return self._symbol_specs_cache.get(symbol)

    def _quote_currency(self, symbol: str) -> Currency | None:
        """The instrument's quote currency from the broker's symbol metadata (its profit currency), or None
        when unavailable — so an asset is denoted in its real currency, not defaulted to the account's."""
        spec = self._symbol_spec(symbol)
        code = getattr(spec, "profit_currency", None) if spec is not None else None
        return Currency(code) if code else None

    def _update_order(self, order: Order) -> None:
        self._client.orders.modify_pending(self._account_id, int(order.id), price=order.limit)

    def _cancel_order(self, order: Order) -> None:
        self._client.orders.cancel_pending(self._account_id, int(order.id))
