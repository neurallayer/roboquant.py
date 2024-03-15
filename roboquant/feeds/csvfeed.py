import csv
import logging
import os
import pathlib
from array import array
from datetime import datetime, time, timezone

from roboquant.event import Bar
from roboquant.feeds.historic import HistoricFeed

logger = logging.getLogger(__name__)


class CSVFeed(HistoricFeed):
    """Use CSV files with historic data as a feed."""

    def __init__(
        self,
        path: str | pathlib.Path,
        columns=None,
        adj_close=False,
        time_offset: str | None = None,
        datetime_fmt: str | None = None,
        endswith=".csv",
        frequency="",
    ):
        super().__init__()
        columns = columns or ["Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]
        self.ohlcv_columns = columns[1:6]
        self.adj_close_column = columns[6] if adj_close else None
        self.date_column = columns[0]
        self.datetime_fmt = datetime_fmt
        self.adj_close = adj_close
        self.freq = frequency
        self.endswith = endswith
        self.time_offset = time.fromisoformat(time_offset) if time_offset is not None else None

        files = self._get_files(path)
        logger.info("located %s files in path %s", len(files), path)
        self._parse_csvfiles(files)  # type: ignore

    def _get_files(self, path):
        if pathlib.Path(path).is_file():
            return [path]

        files = []
        for dirpath, _, filenames in os.walk(path):
            selected_files = [os.path.join(dirpath, f) for f in filenames if f.endswith(self.endswith)]
            files.extend(selected_files)
        return files

    def _get_symbol(self, filename: str):
        """Return the symbol based on the filename"""
        return pathlib.Path(filename).stem.upper()

    def _parse_csvfiles(self, filenames: list[str]):
        adj_close_column = self.adj_close_column
        datetime_fmt = self.datetime_fmt
        ohlcv_columns = self.ohlcv_columns
        date_column = self.date_column
        freq = self.freq
        time_offset = self.time_offset

        for filename in filenames:
            symbol = self._get_symbol(filename)
            with open(filename, encoding="utf8") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    date_str = row[date_column]
                    dt = datetime.strptime(date_str, datetime_fmt) if datetime_fmt else datetime.fromisoformat(date_str)
                    if time_offset:
                        dt = datetime.combine(dt, time_offset)

                    ohlcv = array("f", [float(row[column]) for column in ohlcv_columns])
                    if adj_close_column:
                        adj_close = float(row[adj_close_column])
                        pb = Bar.from_adj_close(symbol, ohlcv, adj_close, freq)
                    else:
                        pb = Bar(symbol, ohlcv, freq)
                    self._add_item(dt.astimezone(timezone.utc), pb)

    @classmethod
    def stooq_us_daily(cls, path):
        """Parse one or more CSV files that meet the stooq format"""
        columns = ["<DATE>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"]

        class StooqCSVFeed(CSVFeed):
            def __init__(self):
                super().__init__(
                    path, columns=columns, time_offset="21:00:00+00:00", datetime_fmt="%Y%m%d", endswith=".txt", frequency="1d"
                )

            def _get_symbol(self, filename: str):
                base = pathlib.Path(filename).stem.upper()
                return base.split(".")[0]

        return StooqCSVFeed()

    @classmethod
    def yahoo(cls, path, frequency="1d"):
        """Parse one or more CSV files that meet the Yahoo Finance format"""
        columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
        return cls(path, columns=columns, adj_close=True, time_offset="21:00:00+00:00", frequency=frequency)
