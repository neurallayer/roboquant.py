import csv
import logging
import os
import pathlib
from array import array
from datetime import datetime, time, timezone

from roboquant.event import Candle
from roboquant.feeds.historicfeed import HistoricFeed

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
        self.columns = columns
        self.time_offset = time_offset
        self.datetime_fmt = datetime_fmt
        self.adj_close = adj_close
        self.endswith = endswith
        self.freq = frequency

        files = self._get_files(path)
        logger.info("located %s files in path %s", len(files), path)

        for file in files:
            self._parse_csvfile(file)  # type: ignore

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

    def _parse_csvfile(self, filename: str):
        adj_close = self.adj_close
        datetime_fmt = self.datetime_fmt
        columns = self.columns or ["Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]
        price_columns = columns[1:7] if adj_close else columns[1:6]
        date_column = columns[0]
        symbol = self._get_symbol(filename)
        freq = self.freq
        t = time.fromisoformat(self.time_offset) if self.time_offset is not None else None

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                dt = (
                    datetime.fromisoformat(row[date_column])  # type: ignore
                    if datetime_fmt is None
                    else datetime.strptime(row[date_column], datetime_fmt)  # type: ignore
                )
                if t:
                    dt = datetime.combine(dt, t)

                prices = array("f", [float(row[column_name]) for column_name in price_columns])
                pb = Candle(symbol, prices, freq) if not adj_close else Candle.from_adj_close(symbol, prices, freq)
                self._add_item(dt.astimezone(timezone.utc), pb)

    @classmethod
    def stooq_us_daily(cls, path):
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
        columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
        return cls(path, columns=columns, adj_close=True, time_offset="21:00:00+00:00", frequency=frequency)
