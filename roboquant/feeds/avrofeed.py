import os.path
from datetime import datetime
from array import array
import time

from fastavro import writer, reader, parse_schema

from roboquant.alpaca.feed import AlpacaHistoricStockFeed
from roboquant.event import Quote, Bar
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.feeds.feedutil import count_events


class AvroFeed(Feed):

    def __init__(self, avro_file) -> None:
        super().__init__()
        self.avro_file = avro_file

        self.schema = {
            "namespace": "org.roboquant.avro.schema",
            "type": "record",
            "name": "PriceItemV2",
            "fields": [
                {"name": "timestamp", "type": "string"},
                {"name": "symbol", "type": "string"},
                {"name": "type", "type": {"type": "enum", "name": "item_type", "symbols": ["BAR", "TRADE", "QUOTE", "BOOK"]}},
                {"name": "values", "type": {"type": "array", "items": "double"}},
                {"name": "meta", "type": ["null", "string"], "default": None},
            ],
        }

    def exists(self):
        return os.path.exists(self.avro_file)

    def play(self, channel: EventChannel):
        t_old = ""
        items = []

        with open(self.avro_file, "rb") as fo:
            for row in reader(fo):
                t = row["timestamp"]  # type: ignore

                if t != t_old:
                    if items:
                        # time_us = int(t_old) // 1_000
                        # dt = datetime.fromtimestamp(time_us // 1_000_000, tz=timezone.utc)
                        # dt = dt.replace(microsecond=time_us % 1_000_000)
                        dt = datetime.fromisoformat(t)
                        event = Event(dt, items)
                        channel.put(event)
                        items = []
                    t_old = t

                price_type = str(row["type"])  # type: ignore
                match (price_type):
                    case "QUOTE":
                        item = Quote(row["symbol"], array("f", row["values"]))  # type: ignore
                        items.append(item)
                    case "BAR":
                        item = Bar(row["symbol"], array("f", row["values"]), row["other"])  # type: ignore
                        items.append(item)
                    case _:
                        raise ValueError(f"Unsupported priceItem type={price_type}")

    def record(self, feed: Feed, timeframe=None):
        schema = parse_schema(self.schema)
        channel = feed.play_background(timeframe)
        records = []
        while event := channel.get():
            t = event.time.isoformat()
            for item in event.items:
                if isinstance(item, Quote):
                    data = {"timestamp": t, "type": "QUOTE", "symbol": item.symbol, "values": list(item.data)}
                    records.append(data)
                if isinstance(item, Bar):
                    data = {
                        "timestamp": t,
                        "type": "BAR",
                        "symbol": item.symbol,
                        "values": list(item.ohlcv),
                        "meta": item.frequency,
                    }
                    records.append(data)

        with open(self.avro_file, "wb") as out:

            writer(out, schema, records)

    def __str__(self) -> str:
        return f"AvroFeed(path={self.avro_file})"


if __name__ == "__main__":

    avroFeed = AvroFeed("/tmp/test.avro")
    if not avroFeed.exists():
        alpaca_feed = AlpacaHistoricStockFeed()
        alpaca_feed.retrieve_quotes("AAPL", start="2024-05-24T20:00:00Z")
        avroFeed.record(alpaca_feed)

    start = time.time()
    print(count_events(avroFeed), time.time() - start)
    # print_feed_items(feed)
