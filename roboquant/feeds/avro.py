import logging
import os.path
from datetime import datetime
from array import array

from fastavro import writer, reader, parse_schema, block_reader
# from fastavro._read_py import block_reader

from roboquant.event import Quote, Bar, Trade
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.asset import deserialize_to_asset
from roboquant.timeframe import Timeframe

logger = logging.getLogger(__name__)


class AvroFeed(Feed):
    """Feed that uses Avro files to store historic prices. Supports `Quote`, `Trade` and `Bar` prices.
    Besides playback, there is also functionality to record another feed into an `AvroFeed`.
    """

    _schema = {
        "namespace": "org.roboquant.avro.schema",
        "type": "record",
        "name": "PriceItemV2",
        "fields": [
            {"name": "timestamp", "type": "string"},
            {"name": "asset", "type": "string"},
            {"name": "type", "type": {"type": "enum", "name": "item_type", "symbols": ["BAR", "TRADE", "QUOTE", "BOOK"]}},
            {"name": "values", "type": {"type": "array", "items": "double"}},
            {"name": "meta", "type": ["null", "string"], "default": None},
        ],
    }

    def __init__(self, avro_file) -> None:
        super().__init__()
        self.avro_file = avro_file
        logger.info("avro feed file=%s", avro_file)

    def exists(self):
        """Check if the avro file exists"""
        return os.path.exists(self.avro_file)

     # typing: ignore
    def __index(self):
        """Create an index on the date column and return it"""
        result = []
        if self.exists():
            with open(self.avro_file, "rb") as fo:
                for block in block_reader(fo):
                    for row in block:
                        t = row["timestamp"]  # type: ignore
                        result.append(datetime.fromisoformat(t))
        return result

    def play(self, channel: EventChannel):
        t_old = ""
        items = []
        start_time = channel.timeframe.start.isoformat() if channel.timeframe else "1900-01-01T00:00:00Z"

        with open(self.avro_file, "rb") as fo:
            for row in reader(fo):
                t = row["timestamp"]  # type: ignore
                if t < start_time:
                    continue

                if t != t_old:
                    if items:
                        dt = datetime.fromisoformat(t)
                        event = Event(dt, items)
                        channel.put(event)
                        items = []
                    t_old = t

                price_type = str(row["type"])  # type: ignore
                asset = deserialize_to_asset(str(row["asset"]))  # type: ignore
                match price_type:
                    case "QUOTE":
                        item = Quote(asset, array("f", row["values"]))  # type: ignore
                        items.append(item)
                    case "BAR":
                        item = Bar(asset, array("f", row["values"]), row["meta"])  # type: ignore
                        items.append(item)
                    case "TRADE":
                        prices = row["values"]  # type: ignore
                        item = Trade(asset, prices[0], prices[1])  # type: ignore
                        items.append(item)
                    case _:
                        raise ValueError(f"Unsupported priceItem type={price_type}")

    def record(self, feed: Feed, timeframe: Timeframe | None=None, append: bool=False, batch_size: int=10_000):
        """Record another feed into an Avro file. It supports a mix of `Quote`, `Trade`, and `Bar` prices.
        Later you can then use this Avro file as a feed to play back the data.
        """

        schema = parse_schema(AvroFeed._schema)
        channel = feed.play_background(timeframe)

        if not append and self.exists():
            os.remove(self.avro_file)

        with open(self.avro_file, "a+b") as out:
            records = []
            while event := channel.get():
                t = event.time.isoformat()
                for item in event.items:
                    asset_str = item.asset.serialize()
                    match item:
                        case Quote():
                            data = {"timestamp": t, "type": "QUOTE", "asset": asset_str, "values": list(item.data)}
                            records.append(data)
                        case Trade():
                            data = {
                                "timestamp": t,
                                "type": "TRADE",
                                "asset": asset_str,
                                "values": [item.trade_price, item.trade_volume],
                            }
                            records.append(data)
                        case Bar():
                            data = {
                                "timestamp": t,
                                "type": "BAR",
                                "asset": asset_str,
                                "values": list(item.ohlcv),
                                "meta": item.frequency,
                            }
                            records.append(data)

                if len(records) > batch_size:
                    writer(out, schema, records)
                    records = []

            if records:
                writer(out, schema, records)

    def __repr__(self) -> str:
        return f"AvroFeed(path={self.avro_file})"
