import threading
import time

from alpaca.data.live.crypto import CryptoDataStream

from roboquant.config import Config
from roboquant.event import Event, Trade
from roboquant.feeds.eventchannel import EventChannel

from roboquant.feeds.feed import Feed


class AlpacaLiveFeed(Feed):

    def __init__(self) -> None:
        super().__init__()
        config = Config()
        api_key = config.get("alpaca.public.key")
        secret_key = config.get("alpaca.secret.key")
        self.stream = CryptoDataStream(api_key, secret_key)
        thread = threading.Thread(None, self.stream.run, daemon=True)
        thread.start()
        # print("running", flush=True)
        self._channel = None

    def play(self, channel: EventChannel):
        self._channel = channel
        while not channel.is_closed:
            time.sleep(1)
        self._channel = None

    async def handle_trades(self, data):
        print(data)
        if self._channel:
            item = Trade(data.symbol, data.price, data.size)
            event = Event(data["timestamp"], [item])
            self._channel.put(event)

    def subscribe(self, *symbols: str):
        self.stream.subscribe_trades(self.handle_trades, *symbols)


def run():
    feed = AlpacaLiveFeed()
    feed.subscribe("BTC/USD", "ETH/USD")
    channel = feed.play_background()
    while event := channel.get(60.0):
        print(event)


if __name__ == "__main__":
    run()
