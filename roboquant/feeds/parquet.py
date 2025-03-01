import logging
import os.path
from array import array
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from roboquant.event import Quote, Bar, Trade
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.asset import deserialize_to_asset, Asset
from roboquant.timeframe import Timeframe

logger = logging.getLogger(__name__)


class ParquetFeed(Feed):
    """PriceItems stored in Parquet Files"""

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
        """Check if the parquet file exists"""
        return os.path.exists(self.parquet_path)

    def play(self, channel: EventChannel):
        # pylint: disable=too-many-locals
        dataset = pq.ParquetFile(self.parquet_path)
        last_time: Any = None
        items = []

        row_group_indexes = self.__get_row_group_indexes(channel.timeframe)

        for batch in dataset.iter_batches(row_groups=row_group_indexes):
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

                asset = deserialize_to_asset(a.as_py())
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

        # any remainders
        if items:
            now = last_time.as_py()
            event = Event(now, items)
            channel.put(event)

    def __get_row_group_indexes(self, timeframe: Timeframe | None) -> Iterable[int]:
        md = pq.read_metadata(self.parquet_path)
        if not timeframe:
            return range(0, md.num_row_groups)

        start: int | None = None
        for idx in range(md.num_row_groups):
            stat = md.row_group(idx).column(0).statistics
            if start is None and stat.max >= timeframe.start:
                start = idx
            if stat.min > timeframe.end:
                return range(start, idx)  # type: ignore

        return [] if start is None else range(start, md.num_row_groups)

    def timeframe(self) -> Timeframe:
        """Return the timeframe of this feed, if the feed is empty it will return an empty timeframe"""
        d = pq.read_metadata(self.parquet_path).to_dict()
        if d["row_groups"]:
            start = d["row_groups"][0]["columns"][0]["statistics"]["min"]
            end = d["row_groups"][-1]["columns"][0]["statistics"]["max"]
            tf = Timeframe(start, end, True)
            return tf
        return Timeframe.EMPTY

    def assets(self) -> list[Asset]:
        """return the list of unique assets available in this feed"""
        if not self.exists():
            return []

        result_table = pq.read_table(self.parquet_path, columns=["asset"], schema=ParquetFeed.__schema)
        assets_list = result_table["asset"].to_pylist()
        assets_set = set(assets_list)
        return list({deserialize_to_asset(s) for s in assets_set})

    def meta(self):
        """Return the metadata of the parquet file"""
        return pq.read_metadata(self.parquet_path)

    def __repr__(self) -> str:
        return f"ParquetFeed(path={self.parquet_path})"

    def record(self, feed: Feed, timeframe: Timeframe | None = None, row_group_size: int=10_000):
        """Record a feed to a parquet file so it can be replayed later on"""

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
