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
    """PriceItems stored in a single Parquet file, supports a mix of `Bar`, `Trade`, and `Quote` price-items.
    Parquet files provide a good balance between speed, memory-size and disk-size, making it a great option to store
    historic market data for back testing.

    Use the `record` method to move data from any other feed into a `ParquetFeed`.
    """

    __schema = pa.schema(
        [
            pa.field("time", pa.timestamp("us", tz="UTC"), False),
            pa.field("asset", pa.string(), False),
            pa.field("type", pa.uint8(), False),
            pa.field("prices", pa.list_(pa.float32()), False),
            pa.field("freq", pa.string(), True),  # only used for Bars
        ]
    )

    def __init__(self, parquet_path) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        logger.info("parquet feed path=%s", parquet_path)

    @staticmethod
    def us_stocks_10():
        """Return a ParquetFeed with the market data for 10 popular S&P 500 companies for 10 years.
        This is included for demo purposes and should not be relied upon for serious back testing.
        """
        path = os.path.join(os.path.dirname(__file__), 'resources', 'us10.parquet')
        return ParquetFeed(path)

    def exists(self) -> bool:
        """Check if the parquet file exists"""
        return os.path.exists(self.parquet_path)

    def play(self, timeframe: Timeframe | None = None):
        # pylint: disable=too-many-locals
        with pq.ParquetFile(self.parquet_path) as dataset:
            last_time: Any = None
            items = []

            row_group_indexes = self.__get_row_group_indexes(timeframe)

            for batch in dataset.iter_batches(row_groups=row_group_indexes):
                times = batch.column("time")
                assets = batch.column("asset")
                prices = batch.column("prices")
                types = batch.column("type")
                freqs = batch.column("freq")
                for n, a, p, t, f in zip(times, assets, prices, types, freqs):
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
                            item = Bar(asset, array("f", p.as_py()), f.as_py())
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
        """
        Determines the row group indexes in a Parquet file that fall within a specified timeframe.

        This method reads the metadata of a Parquet file and identifies the row groups whose
        timestamps overlap with the given timeframe. If no timeframe is provided, it returns
        all row group indexes.

        Args:
            timeframe (Timeframe | None): The timeframe to filter row groups. If None, all
            row groups are included.

        Returns:
            Iterable[int]: A range object or list of integers representing the indexes of
            the row groups that match the specified timeframe. Returns an empty list if no
            row groups match the timeframe.
        """
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
        """Return the timeframe of this feed. If the feed is empty, it will return an empty timeframe"""
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

    def record(self, feed: Feed, timeframe: Timeframe | None = None, row_group_size: int = 10_000, priceitem_filter = None):
        """
        Records a feed to a Parquet file for later replay.

        This method processes events from the provided feed and writes them to a Parquet file.
        Each event is serialized into a specific format based on its type (Quote, Bar, or Trade).
        The data is written in batches to optimize performance. As soon as the number of items is equal or greater
        than the `row_group_size`, it is written as a batch to disk.

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
                    if priceitem_filter and not priceitem_filter(item):
                        continue
                    asset_str = item.asset.serialize()
                    match item:
                        case Quote():
                            items.append({"time": t, "type": 1, "asset": asset_str, "prices": item.data.tolist()})
                        case Bar():
                            items.append(
                                {
                                    "time": t,
                                    "type": 2,
                                    "asset": asset_str,
                                    "prices": item.ohlcv.tolist(),
                                    "freq": item.frequency,
                                }
                            )
                        case Trade():
                            items.append(
                                {"time": t, "type": 3, "asset": asset_str, "prices": [item.trade_price, item.trade_volume]}
                            )

                if len(items) >= row_group_size:
                    batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
                    writer.write_batch(batch)
                    items = []

            # Check for remaining items and write to disk
            if items:
                batch = pa.RecordBatch.from_pylist(items, schema=ParquetFeed.__schema)
                writer.write_batch(batch)
