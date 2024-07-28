import logging
import os.path

from array import array
from datetime import datetime
from typing import Literal

import duckdb

from roboquant.asset import Asset
from roboquant.event import Bar, PriceItem, Quote
from roboquant.event import Event
from roboquant.timeframe import EMPTY_TIMEFRAME, Timeframe
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed

logger = logging.getLogger(__name__)


class DuckDBFeed(Feed):
    """SQLFeed supports recording bars or quotes from other feeds and play them back at a later moment in time.
    It is also possible to append values to an existing database.
    """

    # Used SQL statements in this class
    _sql_select_all = "SELECT * from prices order by Date"
    _sql_select_by_date = "SELECT * from prices where Date >= ? and Date <= ? order by Date"
    _sql_select_timeframe = "SELECT min(Date), max(Date) from prices"
    _sql_count_items = "SELECT count(*) from prices"
    _sql_select_assets = "SELECT DISTINCT asset from prices"
    _sql_drop_table = "DROP TABLE IF EXISTS prices"
    _sql_create_bar_table = """CREATE TABLE IF NOT EXISTS prices(
        date TIMESTAMPTZ, asset VARCHAR, open REAL, high REAL, low REAL, close REAL, volume REAL, frequency VARCHAR)"""
    _sql_create_quote_table = """CREATE TABLE IF NOT EXISTS prices(
        date TIMESTAMPTZ, asset VARCHAR, ap REAL, av REAL, bp REAL, bv REAL)"""
    _sql_insert_bar = "INSERT into prices VALUES(?,?,?,?,?,?,?,?)"
    _sql_insert_quote = "INSERT into prices VALUES(?,?,?,?,?,?)"
    _sql_create_index = "CREATE INDEX IF NOT EXISTS date_idx ON prices(date)"

    def __init__(self, db_file, price_type: Literal["bar", "quote"] = "bar") -> None:
        super().__init__()
        self.db_file = db_file
        self.price_type = price_type
        self.is_bar = price_type == "bar"

    def exists(self):
        return os.path.exists(self.db_file)

    def create_index(self):
        con = duckdb.connect(self.db_file)
        con.execute(DuckDBFeed._sql_create_index)
        con.commit()

    def number_items(self):
        con = duckdb.connect(self.db_file)
        result = con.execute(DuckDBFeed._sql_count_items).fetchall()
        con.commit()
        row = result[0]
        return row[0]

    def timeframe(self):
        con = duckdb.connect(self.db_file)
        con.sql("set TimeZone = 'UTC'")
        result = con.execute(DuckDBFeed._sql_select_timeframe).fetchall()
        con.commit()
        row = result[0]
        if row[0]:
            return Timeframe(row[0], row[1], True)
        return EMPTY_TIMEFRAME

    def assets(self):
        con = duckdb.connect(self.db_file)
        result = con.execute(DuckDBFeed._sql_select_assets).fetchall()
        con.commit()
        assets = {Asset.deserialize(columns[0]) for columns in result}
        return assets

    def get_item(self, row) -> PriceItem:
        if self.is_bar:
            asset_str = row[1]
            asset = Asset.deserialize(asset_str)
            prices = row[2:7]
            freq = row[7]
            return Bar(asset, array("f", prices), freq)

        asset = row[1]
        data = row[2:6]
        return Quote(asset, array("f", data))

    def play(self, channel: EventChannel):
        con = duckdb.connect(self.db_file)
        con.sql("set TimeZone = 'UTC'")
        t_old = datetime.fromisoformat("1900-01-01T00:00:00Z")
        items = []
        tf = channel.timeframe
        result = (
            con.execute(DuckDBFeed._sql_select_by_date, [tf.start, tf.end]) if tf else con.execute(DuckDBFeed._sql_select_all)
        )

        while rows := result.fetchmany(100):
            for row in rows:
                t = row[0]
                assert t >= t_old, f"{t} t_old"
                if t != t_old:
                    if items:
                        event = Event(t_old, items)
                        channel.put(event)
                        items = []
                    t_old = t

                item = self.get_item(row)
                items.append(item)

        # are there leftovers
        if items:
            event = Event(t_old, items)
            channel.put(event)

        con.commit()

    def record(self, feed: Feed, timeframe=None, append=False, batch_size=10_000):
        """Record another feed into this SQLite database.
        It only supports Bars and Quotes"""
        con = duckdb.connect(self.db_file)

        create_sql = DuckDBFeed._sql_create_bar_table if self.is_bar else DuckDBFeed._sql_create_quote_table
        insert_sql = DuckDBFeed._sql_insert_bar if self.is_bar else DuckDBFeed._sql_insert_quote

        if not append:
            con.execute(DuckDBFeed._sql_drop_table)

        con.execute(create_sql)
        price_type = self.price_type

        channel = feed.play_background(timeframe)
        data = []
        con.sql("START TRANSACTION;")
        while event := channel.get():
            t = event.time
            for item in event.items:
                if isinstance(item, Bar) and price_type == "bar":
                    elem = (t.isoformat(), item.asset.serialize(), *item.ohlcv, item.frequency)
                    data.append(elem)
                elif isinstance(item, Quote) and price_type == "quote":
                    elem = (t.isoformat(), item.asset.serialize(), *item.data)
                    data.append(elem)
            if len(data) >= batch_size:
                print("start")
                con.executemany(insert_sql, data)
                print("end")

        if data:
            con.executemany(insert_sql, data)

        con.commit()
        con.close()

    def __repr__(self) -> str:
        return f"DuckDBFeed(timeframe={self.timeframe()} items={self.number_items()} assets={len(self.assets())})"


if __name__ == "__main__":
    qdbfeed = DuckDBFeed("/tmp/test.db", "quote")

    print("The retrieval of historical data will take some time....")
    alpaca_feed = AlpacaHistoricStockFeed()
    alpaca_feed.retrieve_quotes("AAPL", start="2024-05-09T18:00:00Z", end="2024-05-09T19:00:00Z")
    print(alpaca_feed)

    # store it for later use
    qdbfeed.record(alpaca_feed)
    print(qdbfeed)
    print(qdbfeed.assets())

    tf = Timeframe.fromisoformat("2024-05-09T18:00:00Z", "2024-05-09T18:01:00Z", True)
    print(qdbfeed.count_events(tf))
