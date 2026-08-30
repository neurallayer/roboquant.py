import logging
import re
from array import array
from datetime import datetime, timezone
from typing import Any, Self, override

from tickerall import Tickerall, TickerallStream
from tickerall.types import BrokerName, TerminalType, Timeframe

from roboquant.common.asset import Asset, Currency, Forex, USD
from roboquant.common.event import Bar, Event, Quote
from roboquant.common.timeframe import utcnow
from roboquant.feeds.in_memory_feed import InMemoryFeed
from roboquant.feeds.livefeed import LiveFeed

logger = logging.getLogger(__name__)


def _to_asset(symbol: str, quote_currency: Currency | None = None, fallback_currency: Currency = USD) -> Asset:
    """Map a MetaTrader symbol to a roboquant `Forex` asset.

    The asset's currency is the instrument's quote currency. It is taken from `quote_currency` when the
    caller resolved it from the broker's symbol metadata (see `_SymbolCurrency`); this is authoritative and
    is preferred over any inference. When it is not available, the quote currency is inferred from a standard
    6-letter pair (optionally with a broker suffix, e.g. `EURUSDm` -> `USD`), and only when the symbol is not
    a recognizable pair does it fall back to `fallback_currency` (never silently the account currency for a
    symbol whose real currency is known).
    """
    if quote_currency is not None:
        return Forex(symbol, quote_currency)
    core = re.sub(r"[^A-Za-z]", "", symbol).upper()
    # strip a single trailing broker suffix letter that leaves a 6-letter pair (e.g. EURUSDm -> EURUSD)
    if len(core) == 7:
        core = core[:6]
    if len(core) == 6:
        return Forex(symbol, Currency(core[3:6]))
    return Forex(symbol, fallback_currency)


class _SymbolCurrency:
    """Resolves a symbol's quote currency from the broker's symbol metadata, lazily and cached.

    roboquant denotes each asset in its quote currency, and an asset's identity includes that currency, so
    the broker and the feeds must agree on it. The broker exposes each symbol's `profit_currency` (the quote
    currency) on its symbol spec; this loads that map once (one `symbol_specs` call) and serves it to both,
    so a position and a price event for the same symbol resolve to the same asset. Returns `None` when the
    symbol has no spec currency (e.g. an MT4 account, whose spec list is empty), leaving `_to_asset` to fall
    back to the pair heuristic.
    """

    def __init__(self, client: Tickerall, account_id: str) -> None:
        self._client = client
        self._account_id = account_id
        self._cache: dict[str, Currency] | None = None

    def get(self, symbol: str) -> Currency | None:
        if self._cache is None:
            self._cache = {}
            try:
                for spec in self._client.accounts.symbol_specs(self._account_id):
                    code = getattr(spec, "profit_currency", None)
                    if spec.name and code:
                        self._cache[spec.name] = Currency(code)
            except Exception:
                logger.warning("failed to load symbol specs for currency resolution", exc_info=True)
        return self._cache.get(symbol)


def _parse_tick_time(value: str) -> datetime:
    """Parse a tick timestamp (an ISO-8601 string like `2026-07-30T07:26:26.000Z`) into a datetime."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return utcnow()


def _require_tickerall_account_id(account_id: str) -> None:
    """Guard the common mix-up of passing a broker account NUMBER where a TickerAll account id is
    expected. A TickerAll id is a cuid (a 'c' followed by letters and digits); a broker login is all
    digits — fail fast with the fix instead of a later, opaque "Broker account not found"."""
    if account_id and account_id.isdigit():
        raise ValueError(
            f"account_id={account_id!r} looks like a broker account NUMBER, not a TickerAll account id "
            "(a TickerAll id is a cuid, not a number). Connect by MetaTrader credentials with "
            ".connect(api_key, broker=..., server=..., account=..., password=...), or pass the id from "
            "client.sessions.start(...).account_id."
        )


class TickerAllLiveFeed(LiveFeed):
    """Stream live bid/ask ticks for a TickerAll broker account as roboquant `Quote` price-items.

    Built on the official `tickerall` Python SDK. Each tick is published as an `Event` holding a single
    `Quote`. Subscribe to one or more symbols with `subscribe`; call `close` to stop the stream.

    Construct it either from an already-connected `account_id` (e.g. `broker.account_id`, which reuses a
    session the broker already opened) or, for a data-only setup with no broker, from MetaTrader
    credentials with `TickerAllLiveFeed.connect(...)` — see `connect` and `TickerAllBroker`.

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
        self._stream : TickerallStream | None = None
        self._subscribed: set[str] = set()
        # Resolved lazily (a session opened via `connect` sets `_account_id` after __init__).
        self._symbol_currency: _SymbolCurrency | None = None
        # True only when this feed opened the session itself (via `connect`); a session passed in by
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
        """Connect a MetaTrader account by its credentials and return a live feed bound to it.

        Does the session-start step for you (via the SDK's `sessions.keep_alive`, which re-arms the
        account if it goes cold), so you go straight from MetaTrader credentials to a live feed; `close`
        ends the session it opened. When you already have a broker, prefer
        `TickerAllLiveFeed(api_key, broker.account_id)` so the account's session is opened only once.

        Args: as `TickerAllBroker.connect`.
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
    def account_id(self) -> str:
        """The id of the connected TickerAll broker account this feed streams."""
        return self._account_id

    def _quote_currency(self, symbol: str) -> Currency | None:
        """The instrument's quote currency from broker metadata, or None to fall back to inference."""
        if self._symbol_currency is None:
            self._symbol_currency = _SymbolCurrency(self._client, self._account_id)
        return self._symbol_currency.get(symbol)

    def subscribe(self, *symbols: str) -> None:
        """Subscribe to live ticks for the given symbols. Can be called more than once to add symbols."""
        if not symbols:
            return

        if self._stream is None:
            self._stream = self._client.stream.connect()
            self._stream.on("tick", self.__on_tick)
        self._stream.subscribe_ticks(self._account_id, list(symbols))
        self._subscribed.update(symbols)
        # Pre-warm the currency cache (one metadata call) so tick handling never blocks on it.
        self._quote_currency(next(iter(symbols)))

    @override
    def assets(self) -> list[Asset]:
        """The assets subscribed so far on this live feed."""
        return [_to_asset(s, self._quote_currency(s)) for s in sorted(self._subscribed)]

    def __on_tick(self, ev: Any) -> None:
        if ev.symbol is None or ev.bid is None or ev.ask is None:
            return
        asset = self.get_asset(ev.symbol)
        if asset is None:
            asset = _to_asset(ev.symbol, self._quote_currency(ev.symbol))
            self.register(ev.symbol, asset)
        # roboquant Quote data layout: [ask-price, ask-volume, bid-price, bid-volume]. A MetaTrader tick
        # carries no size, so the volumes are left at 0.0 (only the prices are meaningful).
        quote = Quote(asset, array("f", [float(ev.ask), 0.0, float(ev.bid), 0.0]))
        self._put(Event(_parse_tick_time(ev.timestamp), [quote]))

    def close(self) -> None:
        """Close the live feed and its websocket. If it was created with `connect`, this also ends the
        session it opened; a session passed in by `account_id` is left running. Always closes the
        underlying SDK client."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._owns_session and self._account_id:
            try:
                self._client.sessions.end(self._account_id)
            except Exception:
                logger.warning("failed to end TickerAll session %s on close", self._account_id, exc_info=True)
        self._client.close()


class TickerAllHistoricFeed(InMemoryFeed):
    """Load historic OHLC candles for a TickerAll broker account as roboquant `Bar` price-items.

    Built on the official `tickerall` Python SDK. Call `retrieve` for one or more symbols; the bars are kept
    in memory and replayed like any other historic feed (so it can drive a back test).

    Construct it either from an already-connected `account_id` (e.g. `broker.account_id`, which reuses a
    session the broker already opened) or, for a data-only setup with no broker, from MetaTrader
    credentials with `TickerAllHistoricFeed.connect(...)` — see `connect` and `TickerAllBroker`.

    Args:
        api_key: the TickerAll api key.
        account_id: the id of an already-connected TickerAll broker account (see `connect` to open one
            from MetaTrader credentials instead).
        base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api.tickerall.com") -> None:
        super().__init__()
        _require_tickerall_account_id(account_id)
        self._account_id : str = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)
        # Resolved lazily (a session opened via `connect` sets `_account_id` after __init__).
        self._symbol_currency: _SymbolCurrency | None = None
        # True only when this feed opened the session itself (via `connect`); a session passed in by
        # account_id belongs to the caller and is never ended on `close`.
        self._owns_session : bool = False

    def _quote_currency(self, symbol: str) -> Currency | None:
        """The instrument's quote currency from broker metadata, or None to fall back to inference."""
        if self._symbol_currency is None:
            self._symbol_currency = _SymbolCurrency(self._client, self._account_id)
        return self._symbol_currency.get(symbol)

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
        """Connect a MetaTrader account by its credentials and return a historic feed bound to it.

        Does the session-start step for you (via the SDK's `sessions.keep_alive`), so you go straight
        from MetaTrader credentials to a candle feed; `close` ends the session it opened. When you already
        have a broker, prefer `TickerAllHistoricFeed(api_key, broker.account_id)` so the account's session
        is opened only once.

        Args: as `TickerAllBroker.connect`.
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
    def account_id(self) -> str:
        """The id of the connected TickerAll broker account this feed loads candles for."""
        return self._account_id

    def retrieve(self, *symbols: str, timeframe: Timeframe = "H1", hours: int = 168) -> None:
        """Retrieve candles for the given symbols at `timeframe` (e.g. `M1`, `H1`, `D1`) over the last `hours`."""
        for symbol in symbols:
            asset = _to_asset(symbol, self._quote_currency(symbol))
            for candle in self._client.candles.get(self._account_id, symbol=symbol, hours=hours, timeframe=timeframe):
                dt = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
                ohlcv = array(
                    "f",
                    [
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.tick_volume or 0.0,
                    ],
                )
                self._add_item(dt, Bar(asset, ohlcv, timeframe))
        self._update()

    def close(self) -> None:
        """Close the feed. If it was created with `connect`, this also ends the session it opened; a
        session passed in by `account_id` is left running. Always closes the underlying SDK client (its
        HTTP connection pool)."""
        if self._owns_session and self._account_id:
            try:
                self._client.sessions.end(self._account_id)
            except Exception:
                logger.warning("failed to end TickerAll session %s on close", self._account_id, exc_info=True)
        self._client.close()
