import logging
import re
from array import array
from datetime import datetime, timezone
from typing import Self

from tickerall import Tickerall
from tickerall.types import BrokerName, TerminalType, Timeframe

from roboquant.common.asset import Asset, Currency, Forex, USD
from roboquant.common.event import Bar, Event, Quote
from roboquant.common.timeframe import utcnow
from roboquant.feeds.in_memory_feed import InMemoryFeed
from roboquant.feeds.livefeed import LiveFeed

logger = logging.getLogger(__name__)


def _to_asset(symbol: str, fallback_currency: Currency = USD) -> Asset:
    """Map a MetaTrader symbol to a roboquant `Forex` asset.

    MetaTrader symbols are typically a standard currency pair, sometimes with a broker suffix (for example
    `EURUSDm`). The quote currency (the second half of the pair) becomes the asset currency, so `EURUSDm`
    is denoted in `USD`. When the symbol is not a recognizable 6-letter pair, `fallback_currency` is used.
    """
    core = re.sub(r"[^A-Za-z]", "", symbol).upper()
    # strip a single trailing broker suffix letter that leaves a 6-letter pair (e.g. EURUSDm -> EURUSD)
    if len(core) == 7:
        core = core[:6]
    if len(core) == 6:
        return Forex(symbol, Currency(core[3:6]))
    return Forex(symbol, fallback_currency)


def _parse_tick_time(value) -> datetime:
    """Parse a tick timestamp (an ISO-8601 string like `2026-07-30T07:26:26.000Z`) into a datetime."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return utcnow()


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
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)
        self._stream = None
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

    def subscribe(self, *symbols: str) -> None:
        """Subscribe to live ticks for the given symbols. Can be called more than once to add symbols."""
        if self._stream is None:
            self._stream = self._client.stream.connect()
            self._stream.on("tick", self._on_tick)
        self._stream.subscribe_ticks(self._account_id, list(symbols))

    def _on_tick(self, ev) -> None:
        if ev.symbol is None or ev.bid is None or ev.ask is None:
            return
        asset = _to_asset(ev.symbol)
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
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)
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
            asset = _to_asset(symbol)
            for candle in self._client.candles.get(self._account_id, symbol=symbol, hours=hours, timeframe=timeframe):
                dt = datetime.fromtimestamp(int(candle.timestamp), tz=timezone.utc)
                ohlcv = array(
                    "f",
                    [
                        float(candle.open),
                        float(candle.high),
                        float(candle.low),
                        float(candle.close),
                        float(candle.tick_volume or 0.0),
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
