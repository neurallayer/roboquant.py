import logging
import re
from array import array
from datetime import datetime, timezone

from tickerall import Tickerall
from tickerall.types import Timeframe

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

    Args:
        api_key: the TickerAll api key.
        account_id: the id of the connected TickerAll broker account.
        base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api.tickerall.com") -> None:
        super().__init__()
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)
        self._stream = None

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
        """Close the live feed, its websocket, and the underlying SDK client."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._client.close()


class TickerAllHistoricFeed(InMemoryFeed):
    """Load historic OHLC candles for a TickerAll broker account as roboquant `Bar` price-items.

    Built on the official `tickerall` Python SDK. Call `retrieve` for one or more symbols; the bars are kept
    in memory and replayed like any other historic feed (so it can drive a back test).

    Args:
        api_key: the TickerAll api key.
        account_id: the id of the connected TickerAll broker account.
        base_url: the TickerAll REST base url, default `https://api.tickerall.com`.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api.tickerall.com") -> None:
        super().__init__()
        self._account_id = account_id
        self._client = Tickerall(api_key=api_key, base_url=base_url)

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
        """Close the underlying SDK client (its HTTP connection pool)."""
        self._client.close()
