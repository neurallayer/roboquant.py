import logging
import os.path
from array import array
import time
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from roboquant.event import Quote, Bar, Trade
from roboquant.event import Event
from roboquant.feeds.csvfeed import CSVFeed
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.asset import Asset
from roboquant.feeds.feedutil import count_items
from roboquant.timeframe import EMPTY_TIMEFRAME, Timeframe

logger = logging.getLogger(__name__)


class ParquetFeed(Feed):

    __schema = pa.schema(
        [
            pa.field("time", pa.timestamp("us", tz="UTC"), False),
            pa.field("asset", pa.string(), False),
            pa.field("type", pa.uint8(), False),
            pa.field("prices", pa.list_(pa.float32()), False),
        ]
    )

    def __init__(self, parquet_path) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        logger.info("parquet feed path=%s", parquet_path)

    def exists(self):
        return os.path.exists(self.parquet_path)

    def play(self, channel: EventChannel):
        dataset = pq.ParquetFile(self.parquet_path)
        last_time: Any = None
        items = []
        for batch in dataset.iter_batches():
            times = batch.column("time")
            assets = batch.column("asset")
            prices = batch.column("prices")
            types = batch.column("type")
            for n, a, p, t in zip(times, assets, prices, types):

                if n != last_time:
                    if items:
                        now = last_time.as_py()
                        event = Event(now, items)
                        channel.put(event)
                    last_time = n
                    items = []

                asset = Asset.deserialize(a.as_py())
                if t.as_py() == 1:
                    item = Quote(asset, array("f", p.as_py()))
                    items.append(item)
                if t.as_py() == 2:
                    item = Bar(asset, array("f", p.as_py()))
                    items.append(item)
                if t.as_py() == 3:
                    price, volume = p.as_py()
                    item = Trade(asset, price, volume)
                    items.append(item)

        if items:
            now = last_time.as_py()
            event = Event(now, items)
            channel.put(event)

    def _generator(self, feed: Feed, timeframe: Timeframe | None):
        channel = feed.play_background(timeframe)
        t_old = ""
        items = []
        cnt = 0
        while event := channel.get():
            t = event.time

            if t != t_old and cnt > 10_000:
                if items:
                    batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
                    yield batch
                items = []
                t_old = t
                cnt = 0

            for item in event.items:
                match item:
                    case Quote():
                        cnt += 1
                        items.append({"time": t, "type": 1, "asset": item.asset.serialize(), "prices": item.data.tolist()})
                    case Bar():
                        cnt += 1
                        items.append({"time": t, "type": 2, "asset": item.asset.serialize(), "prices": item.ohlcv.tolist()})
                    case Trade():
                        cnt += 1
                        items.append(
                            {
                                "time": t,
                                "type": 3,
                                "asset": item.asset.serialize(),
                                "prices": [item.trade_price, item.trade_volume],
                            }
                        )

        if items:
            batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
            yield batch

    def timeframe(self) -> Timeframe:
        d = pq.read_metadata(self.parquet_path).to_dict()
        if d["row_groups"]:
            start = d["row_groups"][0]["columns"][0]["statistics"]["min"]
            end = d["row_groups"][-1]["columns"][0]["statistics"]["max"]
            tf = Timeframe(start, end, True)
            return tf
        return EMPTY_TIMEFRAME

    def meta(self):
        return pq.read_metadata(self.parquet_path)

    def record2(self, feed: Feed, timeframe: Timeframe | None = None):
        ds.write_dataset(
            self._generator(feed, timeframe),
            self.parquet_path,
            format="parquet",
            schema=ParquetFeed.__schema,
            existing_data_behavior="overwrite_or_ignore",
        )

    def __repr__(self) -> str:
        return f"ParquetFeed(path={self.parquet_path})"

    def record(self, feed: Feed, timeframe: Timeframe | None = None, row_group_size=10_000):
        with pq.ParquetWriter(self.parquet_path, schema=ParquetFeed.__schema, use_dictionary=True) as writer:
            channel = feed.play_background(timeframe)
            items = []
            while event := channel.get():
                t = event.time

                for item in event.items:
                    asset = item.asset.serialize()
                    match item:
                        case Quote():
                            items.append({"time": t, "type": 1, "asset": asset, "prices": item.data.tolist()})
                        case Bar():
                            items.append({"time": t, "type": 2, "asset": asset, "prices": item.ohlcv.tolist()})
                        case Trade():
                            items.append(
                                {
                                    "time": t,
                                    "type": 3,
                                    "asset": asset,
                                    "prices": [item.trade_price, item.trade_volume],
                                }
                            )

                if len(items) >= row_group_size:
                    batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
                    writer.write_batch(batch)
                    items = []

            if items:
                batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
                writer.write_batch(batch)


if __name__ == "__main__":

    parquet_feed = ParquetFeed("/tmp/us.parquet")

    if not parquet_feed.exists():
        input_feed = CSVFeed.stooq_us_daily("/tmp/us/")
        parquet_feed.record(input_feed)

    print(parquet_feed.timeframe())
    start_time = time.time()
    count_items(parquet_feed)
    print("time=", time.time() - start_time)
