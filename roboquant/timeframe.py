import random
from datetime import datetime, timedelta, timezone
from typing import Any


def utcnow() -> datetime:
    """return the current datetime in the UTC timezone"""
    return datetime.now(timezone.utc)


class Timeframe:
    """A timeframe represents a period in time with a specific start- and end-datetime. Timeframes should not be mutated.

    Internally it stores the start and end times as Python datetime objects with the timezone set to `UTC`.
    """

    __slots__ = "start", "end", "inclusive"

    EMPTY: "Timeframe"
    """represents an empty timeframe, with a start and end time set to the same value"""

    INFINITE: "Timeframe"
    """represents an infinite timeframe, with a start time set to the year 1900 and an end time set to the year 2200"""

    def __init__(self, start: datetime, end: datetime, inclusive=False):
        """
        Create a new timeframe. All datetimes will be stored in the UTC timezone.

        Args:
        - start: start datetime
        - end: end datetime
        - inclusive: should the end datetime be inclusive, default is False
        """
        self.start: datetime = start.astimezone(timezone.utc)
        self.end: datetime = end.astimezone(timezone.utc)
        self.inclusive: bool = inclusive

        assert self.start <= self.end, "start > end"

    @classmethod
    def fromisoformat(cls, start: str, end: str, inclusive=False):
        """Create an instance of Timeframe based on a start- and end-datetime in ISO 8601 format.

        Usage:
            tf1 = Timeframe.fromisoformat("2021-01-01T00:12:00+00:00", "2021-01-02T00:13:00+00:00", True)
            tf2 = Timeframe.fromisoformat("2021-01-01", "2022-01-01", False)
        """
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        return cls(s, e, inclusive)

    def is_empty(self):
        """Return true if this is an empty timeframe"""
        return self.start == self.end and not self.inclusive

    @staticmethod
    def previous(inclusive=False, **kwargs):
        """Convenient method to create a historic timeframe, the kwargs arguments will be passed to the timedelta

        Usage:
            tf = Timeframe.previous(days=365)
        """

        td = timedelta(**kwargs)
        end = datetime.now(timezone.utc)
        start = end - td
        return Timeframe(start, end, inclusive)

    @staticmethod
    def next(inclusive=False, **kwargs):
        """Convenient method to create a future timeframe, the kwargs arguments will be passed to the timedelta

        Usage:
            tf = Timeframe.next(minutes=30)
        """

        td = timedelta(**kwargs)
        start = datetime.now(timezone.utc)
        end = start + td
        return Timeframe(start, end, inclusive)

    def __contains__(self, dt: datetime):
        if self.inclusive:
            return self.start <= dt <= self.end
        return self.start <= dt < self.end

    def __repr__(self):
        if self.is_empty():
            return "EMPTY_TIMEFRAME"

        last_char = "]" if self.inclusive else ">"
        fmt_str = "%Y-%m-%d %H:%M:%S"
        return f"[{self.start.strftime(fmt_str)} ― {self.end.strftime(fmt_str)}{last_char}"

    @property
    def duration(self) -> timedelta:
        """return the duration of this timeframe expressed as timedelta"""
        return self.end - self.start

    def annualize(self, rate: float) -> float:
        """annualize the rate for this timeframe"""

        # at least 1 week is required to calculate annualized return
        if self.duration < timedelta(weeks=1):
            return float("NaN")

        years = timedelta(days=365) / self.duration
        return (1.0 + rate) ** years - 1.0

    def split(self, n: int | timedelta | Any) -> list["Timeframe"]:
        """Split the timeframe in sequential equal parts and return the resulting list of timeframes.
        The parameter `n` can be a number, a timedelta instance or other types like `relativedelta` that support
        datetime calculations.

        The last returned timeframe can be shorter than the provided timedelta.
        """

        period = self.duration / n if isinstance(n, int) else n
        end = self.start
        result = []
        while end < self.end:
            start = end
            end = start + period
            result.append(Timeframe(start, end, False))

        if result:
            last = result[-1]
            last.inclusive = self.inclusive
            last.end = self.end
        return result

    def sample(self, duration: timedelta | Any, n: int = 1) -> list["Timeframe"]:
        """Sample one or more periods of `duration` from this timeframe.
        It can contain duplicates and the resulting timeframes can overlap.
        """

        result = []
        end = self.end - duration
        if end < self.start:
            raise ValueError("sample duration is too large for this timeframe")

        while len(result) < n:
            start = random.uniform(self.start.timestamp(), end.timestamp())
            start_dt = datetime.fromtimestamp(start, timezone.utc)
            tf = Timeframe(start_dt, start_dt + duration)
            result.append(tf)
        return result

    def __eq__(self, other):
        if isinstance(other, Timeframe):
            return self.start == other.start and self.end == other.end and self.inclusive == other.inclusive

        return False


Timeframe.EMPTY = Timeframe.fromisoformat("1900-01-01T00:00:00+00:00", "1900-01-01T00:00:00+00:00", False)
Timeframe.INFINITE = Timeframe.fromisoformat("1900-01-01T00:00:00+00:00", "2200-01-01T00:00:00+00:00", True)
