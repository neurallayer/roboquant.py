from datetime import date, datetime, time, timezone
import csv
from array import array
import logging

from roboquant.event import Candle
from roboquant.feeds.historicfeed import HistoricFeed
from roboquant.config import Config
import requests

logger = logging.getLogger(__name__)


class TiingoHistoricFeed(HistoricFeed):
    """Support the following historical market data:

    - EOD historical stock prices
    - Intraday IEX stock prices
    - Intraday crypto prices
    - Intraday forex prices
    """

    def __init__(self, key: str | None = None):
        super().__init__()
        self.key = key or Config().get("tiingo.key")
        assert self.key, "no Tiingo key found"

    @staticmethod
    def __get_csv_iter(response: requests.Response):
        lines = response.content.decode("utf-8").splitlines()
        rows = iter(csv.reader(lines))
        next(rows, None)  # skip header line.
        return rows

    @staticmethod
    def __get_end_date(end_date: str | None = None) -> str:
        result = date.today().strftime("%Y-%m-%d") if end_date is None else end_date
        return result

    def retrieve_eod_stocks(
        self, *symbols: str, start_date="2010-01-01", end_date: str | None = None, closing_time="21:00:00+00:00"
    ):
        """Retrieve stock EOD historic stock prices for the provided symbols"""

        end_date = self.__get_end_date(end_date)
        columns = "date,adjOpen,adjHigh,adjLow,adjClose,adjVolume"  # retrieve only the adjusted prices
        query = f"startDate={start_date}&endDate={end_date}&format=csv&resampleFreq=daily&token={self.key}&columns={columns}"
        closing_time = time.fromisoformat(closing_time)

        for symbol in symbols:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?{query}"
            logger.debug("eod stock url is %s", url)
            response = requests.get(url)
            if not response.ok:
                logger.warning(f"error symbol={symbol} {response.reason}")
                continue

            rows = self.__get_csv_iter(response)
            found = False
            for row in rows:
                d = datetime.fromisoformat(row[0])
                dt = datetime.combine(d, closing_time)
                ohlcv = [float(p) for p in row[1:6]]
                pb = Candle(symbol, array("f", ohlcv), "1d")
                self._add_item(dt, pb)
                found = True

            if not found:
                logger.warning(f"no data retrieved for symbol={symbol}")

    def retrieve_intraday_iex(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        columns = "date,open,high,low,close,volume"
        query = (
            f"startDate={start_date}&endDate={end_date}&format=csv&resampleFreq={frequency}&token={self.key}&columns={columns}"
        )
        for symbol in symbols:
            url = f"https://api.tiingo.com/iex/{symbol}/prices?{query}"
            logger.debug("intraday iex is %s", url)
            response = requests.get(url)
            if not response.ok:
                logger.warning(f"error symbol={symbol} {response.reason}")
                continue

            rows = self.__get_csv_iter(response)
            for row in rows:
                dt = datetime.fromisoformat(row[0]).astimezone(timezone.utc)
                ohlcv = [float(p) for p in row[1:6]]
                pb = Candle(symbol, array("f", ohlcv), frequency)
                self._add_item(dt, pb)

    def retrieve_intraday_crypto(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        symbols_str = ",".join(symbols)
        query = f"tickers={symbols_str}&startDate={start_date}&endDate={end_date}&resampleFreq={frequency}&token={self.key}"

        url = f"https://api.tiingo.com/tiingo/crypto/prices?{query}"
        logger.debug("intraday crypto url is %s", url)
        response = requests.get(url)
        if not response.ok:
            logger.warning(f"error {response.reason}")
            return

        json = response.json()
        for row in json:
            symbol = row["ticker"].upper()
            for e in row["priceData"]:
                dt = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
                ohlcv = [float(e["open"]), float(e["high"]), float(e["low"]), float(e["close"]), float(e["volume"])]
                pb = Candle(symbol, array("f", ohlcv), frequency)
                self._add_item(dt, pb)

    def retrieve_intraday_fx(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        symbols_str = ",".join(symbols)
        query = f"startDate={start_date}&endDate={end_date}&format=csv&resampleFreq={frequency}&token={self.key}"
        url = f"https://api.tiingo.com/tiingo/fx/{symbols_str}/prices?{query}"

        response = requests.get(url)
        logger.debug("intraday fx url is %s", url)
        if not response.ok:
            logger.warning(f"error {response.reason}")
            return

        rows = self.__get_csv_iter(response)
        for row in rows:
            dt = datetime.fromisoformat(row[0]).astimezone(timezone.utc)
            symbol = row[1].upper()
            ohlcv = [float(p) for p in row[2:6]]
            ohlcv.append(float("nan"))
            pb = Candle(symbol, array("f", ohlcv), frequency)
            self._add_item(dt, pb)
