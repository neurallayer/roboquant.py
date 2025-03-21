import csv
from dataclasses import dataclass
import logging
import os
import pathlib
from array import array
from datetime import datetime, time, timezone

from roboquant.asset import Asset, Stock
from roboquant.event import Bar
from roboquant.feeds.historic import HistoricFeed

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CSVColumns:
    """Define the columns in a CSV file that contains historic market data.
    args:
    - date: the column name for the date
    - open: the column name for the open price
    - high: the column name for the high price
    - low: the column name for the low price
    - close: the column name for the close price
    - volume: the column name for the volume, or None if not available
    - adj_close: the column name for the adjusted close price, or None if not available
    - time: the column name for the time, or None if not available
    """

    date: str = "Date"
    open: str = "Open"
    high: str = "High"
    low: str = "Low"
    close: str = "Close"
    volume: str | None = "Volume"
    adj_close: str | None = "Adj Close"
    time: str | None = None

    def get_ohlcv(self, row: dict[str, str]) -> array:
        """Return an array containing the open, high, low, close, and volume from a row in the CSV file"""

        if self.volume is None:
            data = [row[self.open], row[self.high], row[self.low], row[self.close], "nan"]
        else:
            data = [row[self.open], row[self.high], row[self.low], row[self.close], row[self.volume]]

        return array("f", [float(x) for x in data])


class CSVFeed(HistoricFeed):
    """Use CSV files with historic market data as a feed.
    args:
    - path: the path to the CSV file or directory with CSV files
    - columns: the columns in the CSV file, the default one is for Yahoo Finance
    - time_offset: the time offset to apply to the data, default is None
    - date_fmt: the date format to use, or None if the date is in ISO format
    - time_fmt: the time format to use, or None if the time is in ISO format
    - endswith: the file extension to use to select the files
    - frequency: the frequency of the data, use as part of the `Bar` object but no functional impact
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        columns=CSVColumns(),
        time_offset: str | None = None,
        date_fmt: str | None = None,
        time_fmt: str | None = None,
        endswith=".csv",
        frequency="",
    ):
        super().__init__()
        self.columns = columns
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt
        self.freq = frequency
        self.endswith = endswith
        self.time_offset = time.fromisoformat(time_offset) if time_offset is not None else None

        files = self._get_files(path)
        logger.info("located %s files in path %s", len(files), path)
        self._parse_csvfiles(files)  # type: ignore
        self._update()

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
        return Stock(symbol)

    def _parse_csvfiles(self, filenames: list[str]):
        # pylint: disable=too-many-locals
        get_ohlcv = self.columns.get_ohlcv
        adj_close_column = self.columns.adj_close
        date_fmt = self.date_fmt
        time_fmt = self.time_fmt
        date_column = self.columns.date
        time_column = self.columns.time
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

                    ohlcv = get_ohlcv(row)
                    if adj_close_column:
                        adj_close = float(row[adj_close_column])
                        pb = Bar.from_adj_close(asset, ohlcv, adj_close, freq)
                    else:
                        pb = Bar(asset, ohlcv, freq)

                    self._add_item(dt.astimezone(timezone.utc), pb)

    @classmethod
    def stooq_us_daily(cls, path):
        """Parse one or more CSV files that meet the stooq daily file format"""
        columns = CSVColumns(
            date="<DATE>", open="<OPEN>", high="<HIGH>", low="<LOW>", close="<CLOSE>", volume="<VOL>", adj_close=None
        )

        class StooqDailyFeed(CSVFeed):
            def __init__(self):
                super().__init__(path, columns=columns, time_offset="21:00:00+00:00", endswith=".txt", frequency="1d")

            def _get_asset(self, filename: str):
                base = pathlib.Path(filename).stem
                return Stock(base.split(".")[0].upper())

        return StooqDailyFeed()

    @classmethod
    def stooq_us_intraday(cls, path):
        """Parse one or more CSV files that meet the stooq intraday file format"""
        columns = CSVColumns(
            date="<DATE>",
            open="<OPEN>",
            high="<HIGH>",
            low="<LOW>",
            close="<CLOSE>",
            volume="<VOL>",
            time="<TIME>",
            adj_close=None,
        )

        class StooqIntradayFeed(CSVFeed):
            def __init__(self):
                super().__init__(path, columns=columns, endswith=".txt")

            def _get_asset(self, filename: str):
                base = pathlib.Path(filename).stem
                return Stock(base.split(".")[0].upper())

        return StooqIntradayFeed()

    @classmethod
    def yahoo(cls, path, frequency="1d"):
        """Parse one or more CSV files that meet the Yahoo Finance format"""
        return cls(path, time_offset="21:00:00+00:00", frequency=frequency)
