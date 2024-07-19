import csv
from enum import Enum
import sys
import logging
import os
import pathlib
from array import array
from datetime import datetime, time, timezone

from roboquant.asset import Asset, Stock
from roboquant.event import Bar
from roboquant.feeds.historic import HistoricFeed

logger = logging.getLogger(__name__)


class CSVColumn(str, Enum):

    DATE = 'Date'
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = "Volume"
    ADJ_CLOSE = "Adj CLose"
    TIME = "Time"

    def __repr__(self) -> str:
        return self.value

    @staticmethod
    def merge(d: dict["CSVColumn", str]) -> list[str]:
        return [d.get(e, e.value) for e in CSVColumn]


class CSVFeed(HistoricFeed):
    """Use CSV files with historic data as a feed."""

    def __init__(
        self,
        path: str | pathlib.Path,
        columns=None,
        adj_close=False,
        has_time_column=False,
        time_offset: str | None = None,
        date_fmt: str | None = None,
        time_fmt: str | None = None,
        endswith=".csv",
        frequency=""
    ):
        super().__init__()
        columns = columns or ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Time"]
        self.ohlcv_columns = columns[1:6]
        self.adj_close_column = columns[6] if adj_close else None
        self.date_column = columns[0]
        self.time_column = columns[7] if has_time_column else None
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt
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

    def _get_asset(self, filename: str) -> Asset:
        """Return the symbol based on the filename"""
        symbol = pathlib.Path(filename).stem.upper()
        return Stock(symbol, "USD")

    def _parse_csvfiles(self, filenames: list[str]):
        # pylint: disable=too-many-locals
        adj_close_column = self.adj_close_column
        date_fmt = self.date_fmt
        time_fmt = self.time_fmt
        ohlcv_columns = self.ohlcv_columns
        date_column = self.date_column
        time_column = self.time_column
        freq = self.freq
        time_offset = self.time_offset

        for filename in filenames:
            asset = self._get_asset(filename)
            with open(filename, encoding="utf8") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    date_str = row[date_column]
                    dt = datetime.strptime(date_str, date_fmt) if date_fmt else datetime.fromisoformat(date_str)
                    if time_column:
                        time_str = row[time_column]
                        time_val = datetime.strptime(time_str, time_fmt).time() if time_fmt else time.fromisoformat(time_str)
                        dt = datetime.combine(dt, time_val, timezone.utc)

                    if time_offset:
                        dt = datetime.combine(dt, time_offset)

                    ohlcv = array("f", [float(row[column]) for column in ohlcv_columns])
                    if adj_close_column:
                        adj_close = float(row[adj_close_column])
                        pb = Bar.from_adj_close(asset, ohlcv, adj_close, freq)
                    else:
                        pb = Bar(asset, ohlcv, freq)

                    self._add_item(dt.astimezone(timezone.utc), pb)

    @classmethod
    def stooq_us_daily(cls, path):
        """Parse one or more CSV files that meet the stooq daily file format"""
        columns = ["<DATE>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"]

        class StooqDailyFeed(CSVFeed):
            def __init__(self):
                # from Python 3.11 onwards, we can use the fast standard ISO parsing
                if sys.version_info >= (3, 11):
                    super().__init__(path, columns=columns, time_offset="21:00:00+00:00", endswith=".txt", frequency="1d")
                else:
                    super().__init__(
                        path,
                        columns=columns,
                        time_offset="21:00:00+00:00",
                        date_fmt="%Y%m%d",
                        endswith=".txt",
                        frequency="1d",
                    )

            def _get_asset(self, filename: str):
                base = pathlib.Path(filename).stem
                return Stock(base.split(".")[0].upper(), "USD")

        return StooqDailyFeed()

    @classmethod
    def stooq_us_intraday(cls, path):
        """Parse one or more CSV files that meet the stooq intraday file format"""
        columns = ["<DATE>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>", "", "<TIME>"]

        class StooqIntradayFeed(CSVFeed):
            def __init__(self):
                # from Python 3.11 onwards, we can use the faster standard ISO parsing
                if sys.version_info >= (3, 11):
                    super().__init__(path, columns=columns, has_time_column=True, endswith=".txt")
                else:
                    super().__init__(
                        path,
                        columns=columns,
                        has_time_column=True,
                        date_fmt="%Y%m%d",
                        time_fmt="%H%M%S",
                        endswith=".txt"
                    )

            def _get_asset(self, filename: str):
                base = pathlib.Path(filename).stem
                return Stock(base.split(".")[0].upper(), "USD")

        return StooqIntradayFeed()

    @classmethod
    def yahoo(cls, path, frequency="1d"):
        """Parse one or more CSV files that meet the Yahoo Finance format"""
        columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
        return cls(path, columns=columns, adj_close=True, time_offset="21:00:00+00:00", frequency=frequency)
