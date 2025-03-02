import logging
from array import array
from datetime import timezone
import warnings

import yfinance

from roboquant.asset import Stock
from roboquant.event import Bar
from roboquant.feeds.historic import HistoricFeed

logger = logging.getLogger(__name__)


class YahooFeed(HistoricFeed):
    """A feed using the Yahoo Finance API to retrieve historic market data. By default, it will retrieve daily data, but
    you can also specify a different intervals."""

    def __init__(self, *symbols: str, start_date: str = "2010-01-01", end_date: str | None = None, interval="1d"):
        """
        Create a new YahooFeed instance
        Parameters:
        - symbols: list of symbols to retrieve
        - start_date: the start date of the data to retrieve, default is `2010-01-01`
        - end_date: the end date of the data to retrieve, default is `None` (today)
        - interval: the interval of the data to retrieve, default is `1d` (daily)
        """

        super().__init__()

        # Disable some yfinance warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=DeprecationWarning)

        columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

        for symbol in symbols:
            logger.debug("requesting symbol=%s", symbol)
            df = yfinance.Ticker(symbol).history(
                start=start_date, end=end_date, auto_adjust=False, actions=False, interval=interval, timeout=30
            )[columns]

            assert df is not None

            if len(df) == 0:
                logger.warning("no data retrieved for symbol=%s", symbol)
                continue

            # yFinance one doesn't correct the volume, so we use our own auto-adjust
            self.__auto_adjust(df)

            for t in df.itertuples(index=True):
                dt = t[0].to_pydatetime().astimezone(timezone.utc)
                prices = t[1:6]
                asset = self._get_asset(symbol)
                b = Bar(asset, array("f", prices), interval)
                self._add_item(dt, b)

            logger.info("retrieved symbol=%s items=%s", symbol, len(df))

        self._update()

    def _get_asset(self, symbol: str):
        """Get the asset for the given symbol. The default implementation will return a Stock denoted in USD.
        Subclasses can override this method to provide a different asset type."""
        return Stock(symbol)

    @staticmethod
    def __auto_adjust(df):
        ratio = df["Adj Close"] / df["Close"]
        df["Open"] = df["Open"] * ratio
        df["High"] = df["High"] * ratio
        df["Low"] = df["Low"] * ratio
        df["Close"] = df["Adj Close"]
        df["Volume"] = df["Volume"] / ratio
        df.drop(columns="Adj Close", inplace=True)
