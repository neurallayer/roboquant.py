import logging
from array import array
from datetime import datetime, timezone

import yfinance

from roboquant.event import Candle
from roboquant.feeds.historicfeed import HistoricFeed

logger = logging.getLogger(__name__)


class YahooFeed(HistoricFeed):
    """A feed using the Yahoo Finance API to retrieve historic price data"""

    def __init__(self, *symbols: str, start_date="2010-01-01", end_date: str | None = None, interval="1d"):
        super().__init__()

        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

        for symbol in symbols:
            logger.debug("requesting symbol=%s", symbol)
            df = yfinance.Ticker(symbol).history(
                start=start_date, end=end_date, auto_adjust=False, actions=False, interval=interval
            )[columns]
            df.dropna(inplace=True)

            if len(df) == 0:
                logger.warning("no data retrieved for symbol=%s", symbol)
                continue

            # yFinance one doesn't correct the volume, so we use this one instead
            self.__auto_adjust(df)
            for t in df.itertuples(index=True):
                dt = t[0].to_pydatetime().astimezone(timezone.utc)
                prices = t[1:6]
                candle = Candle(symbol, array("f", prices), interval)
                self._add_item(dt, candle)

            logger.info("retrieved symbol=%s items=%s", symbol, len(df))

    @staticmethod
    def __auto_adjust(df):
        """small routine to apply adj close"""
        ratio = df["Adj Close"] / df["Close"]
        df["Open"] = df["Open"] * ratio
        df["High"] = df["High"] * ratio
        df["Low"] = df["Low"] * ratio
        df["Close"] = df["Adj Close"]
        df["Volume"] = df["Volume"] / ratio
        df.drop(columns="Adj Close", inplace=True)
