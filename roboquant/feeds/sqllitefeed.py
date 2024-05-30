import os.path
import sqlite3
from array import array
from datetime import datetime
from typing import Literal

from roboquant.event import Bar, PriceItem, Quote
from roboquant.event import Event
from roboquant.timeframe import Timeframe
from .eventchannel import EventChannel
from .feed import Feed


class SQLFeed(Feed):
    # Used SQL statements in this class
    _sql_select = "SELECT * from prices order by Date"
    _sql_select_by_date = "SELECT * from prices where Date >= ? and Date <= ? order by Date"
    _sql_select_timeframe = "SELECT min(Date), max(Date) from prices"
    _sql_count_items = "SELECT count(*) from prices"
    _sql_select_symbols = "SELECT DISTINCT symbol from prices"
    _sql_drop_table = "DROP TABLE IF EXISTS prices"
    _sql_create_bar_table = "CREATE TABLE IF NOT EXISTS prices(date, symbol, open, high, low, close, volume, frequency)"
    _sql_create_quote_table = "CREATE TABLE IF NOT EXISTS prices(date, symbol, ap, av, bp, bv)"
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
        con = sqlite3.connect(self.db_file)
        con.execute(SQLFeed._sql_create_index)
        con.commit()

    def items(self):
        con = sqlite3.connect(self.db_file)
        result = con.execute(SQLFeed._sql_count_items).fetchall()
        con.commit()
        row = result[0]
        return row[0]

    def timeframe(self):
        con = sqlite3.connect(self.db_file)
        result = con.execute(SQLFeed._sql_select_timeframe).fetchall()
        con.commit()
        row = result[0]
        tf = Timeframe.fromisoformat(row[0], row[1], True)
        return tf

    def symbols(self):
        con = sqlite3.connect(self.db_file)
        result = con.execute(SQLFeed._sql_select_symbols).fetchall()
        con.commit()
        symbols = [columns[0] for columns in result]
        return symbols

    def get_item(self, row) -> PriceItem:
        if self.is_bar:
            symbol = row[1]
            prices = row[2:7]
            freq = row[7]
            return Bar(symbol, array("f", prices), freq)

        symbol = row[1]
        data = row[2:6]
        return Quote(symbol, array("f", data))

    def play(self, channel: EventChannel):
        con = sqlite3.connect(self.db_file)
        cur = con.cursor()
        cnt = 0
        t_old = ""
        items = []
        tf = channel.timeframe
        result = (
            cur.execute(SQLFeed._sql_select_by_date, [tf.start.isoformat(), tf.end.isoformat()])
            if tf
            else cur.execute(SQLFeed._sql_select)
        )

        for row in result:
            t = row[0]
            assert t >= t_old, f"{t} t_old"
            if t != t_old:
                if items:
                    dt = datetime.fromisoformat(t_old)
                    event = Event(dt, items)
                    channel.put(event)
                    items = []
                t_old = t

            item = self.get_item(row)
            items.append(item)
            cnt += 1
        con.commit()

    def record(self, feed: Feed, timeframe=None, append=False):
        """Record another feed to this SQLite database.
        It only supports Bars and not types of prices."""
        con = sqlite3.connect(self.db_file)
        cur = con.cursor()

        create_sql = SQLFeed._sql_create_bar_table if self.is_bar else SQLFeed._sql_create_quote_table
        insert_sql = SQLFeed._sql_insert_bar if self.is_bar else SQLFeed._sql_insert_quote

        if not append:
            cur.execute(SQLFeed._sql_drop_table)

        cur.execute(create_sql)
        price_type = self.price_type

        channel = feed.play_background(timeframe)
        data = []
        while event := channel.get():
            t = event.time
            for item in event.items:
                if isinstance(item, Bar) and price_type == "bar":
                    elem = (t.isoformat(), item.symbol, *item.ohlcv, item.frequency)
                    data.append(elem)
                elif isinstance(item, Quote) and price_type == "quote":
                    elem = (t.isoformat(), item.symbol, *item.data)
                    data.append(elem)

        cur.executemany(insert_sql, data)
        con.commit()

    def __repr__(self) -> str:
        return f"SQLFeed(timeframe={self.timeframe()} items={self.items()} symbols={len(self.symbols())})"
