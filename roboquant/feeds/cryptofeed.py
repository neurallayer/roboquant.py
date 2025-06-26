import logging
import ccxt
from array import array
from datetime import date, datetime, timezone, timedelta

from roboquant.asset import Asset, Crypto
from roboquant.event import Bar
from roboquant.feeds.historic import HistoricFeed

logger = logging.getLogger(__name__)


class CryptoFeed(HistoricFeed):
    """A feed using the Yahoo Finance to retrieve historic market data. By default, it will retrieve daily data, but
    you can specify a different interval."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        *symbols: str,
        start_date: str | date | datetime = "2020-01-01T00:00:00",
        end_date: str | date | datetime | None = None,
        interval="1d",
    ):
        """
        Create a new CryptoFeed instance
        Parameters:
        - symbols: list of symbols to retrieve
        - start_date: the start date of the data to retrieve, default in `2020-01-01`
        - end_date: the end date of the data to retrieve, default is `None` (today)
        - interval: the interval of the data to retrieve, default is `1d` (daily)
        """

        super().__init__()

        if not exchange.has["fetchOHLCV"]:
            raise ValueError(f"Exchange {exchange} does not support fetching OHLCV data")


        start_date = str(start_date)
        end_date = datetime.fromisoformat(str(end_date)) if end_date else None

        for symbol in symbols:

            try:
                asset = self._get_asset(symbol)
                logger.debug("requesting symbol=%s", symbol)
                done = False
                since = str(start_date)

                while not done:

                    # fetch_ohlcv returns a list of lists, each containing [timestamp, open, high, low, close, volume]
                    rows: list[list[float]] = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=interval,
                        since=exchange.parse8601(since),
                        limit=None,
                    ) # type: ignore

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

                    since = str(dt + timedelta(milliseconds=1))  # move to next second

                    logger.info("retrieved symbol=%s items=%s", symbol, len(rows))
            except Exception as e:  # noqa: E722
                print(e)
                logger.error("Error retrieving symbol=%s", symbol)

        self._update()

    def _get_asset(self, symbol: str) -> Asset:
        """Get the asset for the given symbol. The default implementation will return a Stock denoted in USD.
        Subclasses can override this method to provide a different asset type."""
        return Crypto.from_symbol(symbol)



if __name__ == "__main__":
    # Example usage
    exchange = ccxt.binance()  # or any other exchange supported by ccxt
    feed = CryptoFeed(exchange, "BTC/USDT", "ETH/USDT", start_date="2024-01-01T00:00:00", interval="1d")
    print(feed.assets())
    print("Feed initialized with", len(feed.timeline()), "events")
    print(feed.timeframe())

