import logging
import os.path
from typing import Literal

from questdb.ingress import Sender  # pylint: disable=no-name-in-module

from roboquant.timeframe import Timeframe
from roboquant.event import Bar, Quote, Trade
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed

logger = logging.getLogger(__name__)


class QuestDBFeed(Feed):
    """SQLFeed supports recording bars or quotes from other feeds and play them back at a later moment in time.
    It is also possible to append values to an existing database.
    """

    def __init__(self, price_type: Literal["bar", "quote"] = "bar") -> None:
        super().__init__()
        self.db_file = ""
        self.price_type = price_type
        self.is_bar = price_type == "bar"

    def exists(self):
        return os.path.exists(self.db_file)

    def play(self, channel: EventChannel):
        pass

    def record(self, feed: Feed, timeframe: Timeframe | None = None):
        """Record another feed into this SQLite database.
        It only supports Bars and Quotes"""

        conf = "http::addr=localhost:9000;"
        with Sender.from_conf(conf) as sender:

            channel = feed.play_background(timeframe)
            while event := channel.get():
                t = event.time
                for item in event.items:
                    if isinstance(item, Bar):
                        o, h, l, c, v = item.ohlcv
                        sender.row(
                            "prices",
                            symbols={"asset": item.asset.serialize(), "type": "BAR"},
                            columns={
                                "open": o,
                                "high": h,
                                "low": l,
                                "close": c,
                                "volume": v,
                            },
                            at=t,
                        )

                    elif isinstance(item, Trade):
                        sender.row(
                            "prices",
                            symbols={"asset": item.asset.serialize(), "type": "TRADE"},
                            columns={"price": item.price, "volume": item.volume},
                            at=t,
                        )

                    elif isinstance(item, Quote):
                        sender.row(
                            "prices",
                            symbols={"asset": item.asset.serialize(), "type": "QUOTE"},
                            columns={
                                "ask_price": item.ask_price,
                                "bid_price": item.bid_price,
                                "ask_volume": item.ask_volume,
                                "bid_volume": item.bid_volume,
                            },
                            at=t,
                        )

                sender.flush()

    def __repr__(self) -> str:
        return f"QuestDBFeed(timeframe={self.timeframe()})"


if __name__ == "__main__":
    qdbfeed = QuestDBFeed()

    print("The retrieval of historical data will take some time....")
    alpaca_feed = AlpacaHistoricStockFeed()
    alpaca_feed.retrieve_quotes("AAPL", start="2024-05-09T18:00:00Z", end="2024-05-09T18:05:00Z")
    print(alpaca_feed)

    # store it for later use
    qdbfeed.record(alpaca_feed)
