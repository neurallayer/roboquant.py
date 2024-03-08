import random
from datetime import datetime, timedelta, timezone
from typing import Any


class Timeframe:
    """A timeframe represents a period in time with a specific start- and end-datetime.

    Internally it stores the start and end times as Python datetime objects with the timezone set to UTC.
    """

    __slots__ = "start", "end", "inclusive"

    def __init__(self, start: datetime, end: datetime, inclusive=False):
        """
        Create a new timeframe

        Args:
        - start: start time
        - end: end time
        - inclusive: should the end time be inclusive, default is False
        """
        self.start: datetime = start.astimezone(timezone.utc)
        self.end: datetime = end.astimezone(timezone.utc)
        self.inclusive: bool = inclusive

        assert self.start <= self.end, "start > end"

    @classmethod
    def fromisoformat(cls, start: str, end: str, inclusive=False):
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        return cls(s, e, inclusive)

    @staticmethod
    def empty():
        return Timeframe.fromisoformat("1900-01-01T00:00:00+00:00", "1900-01-01T00:00:00+00:00", False)

    @staticmethod
    def previous(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, inclusive=False):
        """Convenient method to create a historic timeframe"""

        td = timedelta(
            days=days,
            seconds=seconds,
            microseconds=microseconds,
            milliseconds=milliseconds,
            minutes=minutes,
            hours=hours,
            weeks=weeks,
        )
        end = datetime.now(timezone.utc)
        start = end - td
        return Timeframe(start, end, inclusive)

    @staticmethod
    def next(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, inclusive=False):
        """Convenient method to create a future timeframe"""

        td = timedelta(
            days=days,
            seconds=seconds,
            microseconds=microseconds,
            milliseconds=milliseconds,
            minutes=minutes,
            hours=hours,
            weeks=weeks,
        )
        start = datetime.now(timezone.utc)
        end = start + td
        return Timeframe(start, end, inclusive)

    def __contains__(self, time):
        if self.inclusive:
            return self.start <= time <= self.end
        return self.start <= time < self.end

    def __repr__(self):
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
        """Split the timeframe in sequential parts and return the resulting list of timeframes.
        The parameter `n` can be a number or a timedelta instance or other types like relativedelta that support
        datetime calculations.
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
        """Sample one or more periods of `duration` from this timeframe."""

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
