import sqlite3
from array import array
from datetime import datetime

from roboquant.event import Bar
from roboquant.event import Event
from roboquant.timeframe import Timeframe
from .eventchannel import EventChannel
from .feed import Feed


class SQLFeed(Feed):
    # Used SQL statements in this class
    _sql_select = "SELECT * from prices order by Date"
    _sql_select_by_date = "SELECT * from prices where Date >= ? and Date <= ? order by Date"
    _sql_select_timeframe = "SELECT min(Date), max(Date) from prices"
    _sql_select_symbols = "SELECT DISTINCT symbol from prices"
    _sql_drop_table = "DROP TABLE IF EXISTS prices"
    _sql_create_table = "CREATE TABLE IF NOT EXISTS prices(date, symbol, open, high, low, close, volume, frequency)"
    _sql_insert_row = "INSERT into prices VALUES(?,?,?,?,?,?,?,?)"
    _sql_create_index = "CREATE INDEX IF NOT EXISTS date_idx ON prices(date)"

    def __init__(self, db_file) -> None:
        super().__init__()
        self.db_file = db_file

    def create_index(self):
        con = sqlite3.connect(self.db_file)
        con.execute(SQLFeed._sql_create_index)
        con.commit()

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

    def play(self, channel: EventChannel):
        con = sqlite3.connect(self.db_file)
        cur = con.cursor()
        cnt = 0
        t_old = ""
        items = []
        tf = channel.timeframe
        result = cur.execute(SQLFeed._sql_select_by_date, [tf.start, tf.end]) if tf else cur.execute(SQLFeed._sql_select)

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

            symbol = row[1]
            prices = row[2:7]
            freq = row[7]
            pb = Bar(symbol, array('f', prices), freq)
            items.append(pb)
            cnt += 1
        con.commit()

    def record(self, feed: Feed, timeframe=None, append=False):
        """Record another feed to this SQLite database"""
        con = sqlite3.connect(self.db_file)
        cur = con.cursor()

        if not append:
            cur.execute(SQLFeed._sql_drop_table)

        cur.execute(SQLFeed._sql_create_table)

        channel = feed.play_background(timeframe)
        data = []
        while event := channel.get():
            t = event.time
            for item in event.items:
                if isinstance(item, Bar):
                    ohlcv = item.ohlcv
                    elem = (t.isoformat(), item.symbol, *ohlcv, item.frequency)
                    data.append(elem)

        cur.executemany(SQLFeed._sql_insert_row, data)
        con.commit()
