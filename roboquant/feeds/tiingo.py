import csv
import json
import logging
import ssl
import threading
from array import array
from datetime import date, datetime, time, timezone
from time import sleep
from typing import Literal

import requests
import websocket

from roboquant.config import Config
from roboquant.event import Bar
from roboquant.event import Trade, Quote, Event
from roboquant.feeds.historic import HistoricFeed
from roboquant.feeds.live import LiveFeed

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
        self.timeout = 10

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
            response = requests.get(url, timeout=self.timeout)
            if not response.ok:
                logger.warning("error symbol=%s reason=%s", symbol, response.reason)
                continue

            rows = self.__get_csv_iter(response)
            found = False
            for row in rows:
                d = datetime.fromisoformat(row[0])
                dt = datetime.combine(d, closing_time)
                ohlcv = [float(p) for p in row[1:6]]
                pb = Bar(symbol, array("f", ohlcv), "1d")
                self._add_item(dt, pb)
                found = True

            if not found:
                logger.warning("no data retrieved for symbol=%s", symbol)

    def retrieve_intraday_iex(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        columns = "date,open,high,low,close,volume"
        query = (
            f"startDate={start_date}&endDate={end_date}&format=csv&resampleFreq={frequency}&token={self.key}&columns={columns}"
        )
        for symbol in symbols:
            url = f"https://api.tiingo.com/iex/{symbol}/prices?{query}"
            logger.debug("intraday iex is %s", url)
            response = requests.get(url, timeout=self.timeout)
            if not response.ok:
                logger.warning("error symbol=%s reason=%s", symbol, response.reason)
                continue

            rows = self.__get_csv_iter(response)
            for row in rows:
                dt = datetime.fromisoformat(row[0]).astimezone(timezone.utc)
                ohlcv = [float(p) for p in row[1:6]]
                pb = Bar(symbol, array("f", ohlcv), frequency)
                self._add_item(dt, pb)

    def retrieve_intraday_crypto(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        symbols_str = ",".join(symbols)
        query = f"tickers={symbols_str}&startDate={start_date}&endDate={end_date}&resampleFreq={frequency}&token={self.key}"

        url = f"https://api.tiingo.com/tiingo/crypto/prices?{query}"
        logger.debug("intraday crypto url is %s", url)
        response = requests.get(url, timeout=self.timeout)
        if not response.ok:
            logger.warning("error reason=%s", response.reason)
            return

        json_resp = response.json()
        for row in json_resp:
            symbol = row["ticker"].upper()
            for e in row["priceData"]:
                dt = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
                ohlcv = [float(e["open"]), float(e["high"]), float(e["low"]), float(e["close"]), float(e["volume"])]
                pb = Bar(symbol, array("f", ohlcv), frequency)
                self._add_item(dt, pb)

    def retrieve_intraday_fx(self, *symbols: str, start_date="2023-01-01", end_date: str | None = None, frequency="5min"):
        end_date = self.__get_end_date(end_date)
        symbols_str = ",".join(symbols)
        query = f"startDate={start_date}&endDate={end_date}&format=csv&resampleFreq={frequency}&token={self.key}"
        url = f"https://api.tiingo.com/tiingo/fx/{symbols_str}/prices?{query}"

        response = requests.get(url, timeout=self.timeout)
        logger.debug("intraday fx url is %s", url)
        if not response.ok:
            logger.warning("error reason=%s", response.reason)
            return

        rows = self.__get_csv_iter(response)
        for row in rows:
            dt = datetime.fromisoformat(row[0]).astimezone(timezone.utc)
            symbol = row[1].upper()
            ohlcv = [float(p) for p in row[2:6]]
            ohlcv.append(float("nan"))
            pb = Bar(symbol, array("f", ohlcv), frequency)
            self._add_item(dt, pb)


class TiingoLiveFeed(LiveFeed):
    """Subscribe to real-time market data from Tiingo

    There is support for US stocks, crypto and forex. See Tiingo.com for more details.
    """

    def __init__(self, key: str | None = None, market: Literal["crypto", "iex", "fx"] = "crypto"):
        super().__init__()
        self.__key = key or Config().get("tiingo.key")

        url = f"wss://api.tiingo.com/{market}"
        logger.info("Opening websocket url=%s", url)
        self.ws = websocket.WebSocketApp(  # type: ignore
            url, on_message=self._handle_message, on_error=self._handle_error, on_close=self._handle_close  # type: ignore
        )
        kwargs = {"sslopt": {"cert_reqs": ssl.CERT_NONE}}
        wst = threading.Thread(target=self.ws.run_forever, kwargs=kwargs, daemon=True)
        wst.daemon = True
        wst.start()
        sleep(3)
        self._last_time = datetime.fromisoformat("1900-01-01T00:00:00+00:00")

    def _deliver(self, price, now: datetime):
        now = now.astimezone(timezone.utc)
        event = Event(now, [price])
        self.put(event)

    def _handle_message_iex(self, arr):
        if arr[13] == 1:  # inter-market sweep order
            return

        now = datetime.fromisoformat(arr[1])
        if arr[0] == "T":
            price = Trade(arr[3].upper(), arr[9], arr[10])
            self._deliver(price, now)

        elif arr[0] == "Q":
            data = array("f", [arr[7], arr[8], arr[5], arr[4]])
            price = Quote(arr[3].upper(), data)
            self._deliver(price, now)

    def _handle_message_fx(self, arr):
        if arr[0] == "Q":
            now = datetime.fromisoformat(arr[2])
            data = array("f", [arr[7], arr[6], arr[4], arr[3]])
            price = Quote(arr[1].upper(), data)
            self._deliver(price, now)

    def _handle_message_crypto(self, arr):
        now = datetime.fromisoformat(arr[2])
        if arr[0] == "T":
            price = Trade(arr[1].upper(), arr[5], arr[4])
            self._deliver(price, now)

        elif arr[0] == "Q":
            data = array("f", [arr[8], arr[7], arr[5], arr[4]])
            price = Quote(arr[1].upper(), data)
            self._deliver(price, now)

    def _handle_message(self, _, msg):
        data = json.loads(msg)
        logger.debug("received json %s", data)

        if data["messageType"] != "A":
            return

        service = data["service"]
        arr = data["data"]

        match service:
            case "iex":
                self._handle_message_iex(arr)
            case "crypto_data":
                self._handle_message_crypto(arr)
            case "fx":
                self._handle_message_fx(arr)

    @staticmethod
    def _handle_error(_, msg):
        logger.error(msg)

    @staticmethod
    def _handle_close(_, close_status_code, close_msg):
        logger.info("Webchannel closed code=%s msg=%s", close_status_code, close_msg)

    def subscribe(self, *symbols: str, threshold_level=5):
        logger.info("Subscribing to %s", symbols or "all symbols")
        msg = {"eventName": "subscribe", "authorization": self.__key, "eventData": {"thresholdLevel": threshold_level}}
        if len(symbols) > 0:
            msg["eventData"]["tickers"] = symbols

        json_str = json.dumps(msg)
        logger.info("json %s", json_str)
        self.ws.send(json_str)

    def close(self):
        self._last_time = datetime.fromisoformat("1900-01-01").astimezone(timezone.utc)
        self.ws.close()
