import logging
import os.path
from array import array
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from roboquant.event import Quote, Bar, Trade
from roboquant.event import Event
from roboquant.feeds.feed import Feed
from roboquant.asset import deserialize_to_asset, Asset
from roboquant.timeframe import Timeframe

logger = logging.getLogger(__name__)


class ParquetFeed(Feed):
    """PriceItems stored in Parquet files, supports a mix of `Bar`, `Trade`, and `Quote` price-items."""

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

    def play(self, timeframe: Timeframe | None = None):
        # pylint: disable=too-many-locals
        dataset = pq.ParquetFile(self.parquet_path)
        last_time: Any = None
        items = []

        row_group_indexes = self.__get_row_group_indexes(timeframe)

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
                        yield event
                    last_time = n
                    items = []

                asset = deserialize_to_asset(a.as_py())
                match t.as_py():
                    case 1:
                        item = Quote(asset, array("f", p.as_py()))
                        items.append(item)
                    case 2:
                        item = Bar(asset, array("f", p.as_py()))
                        items.append(item)
                    case 3:
                        price, volume = p.as_py()
                        item = Trade(asset, price, volume)
                        items.append(item)
                    case _:
                        logger.warning("Unknown type %s", t.as_py())

        # any remainders
        if items:
            now = last_time.as_py()
            event = Event(now, items)
            yield event

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
        """
        Records a feed to a Parquet file for later replay.

        This method processes events from the provided feed and writes them to a Parquet file.
        Each event is serialized into a specific format based on its type (Quote, Bar, or Trade).
        The data is written in batches to optimize performance.

        Args:
            feed (Feed): The feed containing events to be recorded.
            timeframe (Timeframe | None, optional): The timeframe to filter events. If None, all events are processed.
            row_group_size (int, optional): The number of rows to include in each batch written to the Parquet file.
            Defaults to 10,000.

        Notes:
            - Events are serialized with the following structure:
            - `time`: The timestamp of the event.
            - `type`: The type of the event (1 for Quote, 2 for Bar, 3 for Trade).
            - `asset`: Serialized representation of the asset.
            - `prices`: A list of price-related data specific to the event type.
        """

        with pq.ParquetWriter(self.parquet_path, schema=ParquetFeed.__schema, use_dictionary=True) as writer:
            items = []
            for event in feed.play(timeframe):
                t = event.time

                for item in event.items:
                    if not isinstance(item, (Quote, Bar, Trade)):
                        continue
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
