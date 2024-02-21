from datetime import datetime, timezone
from typing import Literal
import websocket
import ssl
import json
import threading
import time
from array import array
import logging

from roboquant.config import Config
from roboquant.event import Trade, Quote, Event
from roboquant.feeds.feed import Feed
from roboquant.feeds.eventchannel import EventChannel

logger = logging.getLogger(__name__)


class TiingoLiveFeed(Feed):
    """Subscribe to real-time market data from Tiingo

    There is support for US stocks, crypto and forex. See Tiingo.com for more details.
    """

    def __init__(self, key: str | None = None, market: Literal["crypto", "iex", "fx"] = "crypto"):

        self.__key = key or Config().get("tiingo.key")
        self.channel = None

        url = f"wss://api.tiingo.com/{market}"
        logger.info(f"Opening websocket {url}")
        self.ws = websocket.WebSocketApp(  # type: ignore
            url, on_message=self._handle_message, on_error=self._handle_error, on_close=self._handle_close
        )
        kwargs = {"sslopt": {"cert_reqs": ssl.CERT_NONE}}
        wst = threading.Thread(target=self.ws.run_forever, kwargs=kwargs)
        wst.daemon = True
        wst.start()
        time.sleep(3)
        self._last_time = datetime.fromisoformat("1900-01-01T00:00:00+00:00")

    def play(self, channel: EventChannel):
        self.channel = channel
        while not channel.is_closed:
            time.sleep(1)
        self.channel = None

    def _deliver(self, price, now: datetime):
        if self.channel and not self.channel.is_closed:
            t = now.astimezone(timezone.utc)

            # required for crypto times
            if t < self._last_time:
                t = self._last_time
            else:
                self._last_time = t

            event = Event(t, [price])
            self.channel.put(event)

    def _handle_message_iex(self, arr):
        if arr[13] == 1:  # intermarket sweep order
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

    def _handle_error(self, ws, msg):
        logger.error(msg)

    def _handle_close(self, ws, close_status_code, close_msg):
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
