import random
from datetime import datetime, timedelta, timezone
from typing import Any


def utcnow() -> datetime:
    """Return the current datetime in the UTC timezone."""
    return datetime.now(timezone.utc)


class Timeframe:
    """A timeframe represents a period in time with a specific start- and end-datetime. Timeframes should not be mutated.

    Internally it stores the start and end times as Python datetime objects with the timezone set to `UTC`.
    """

    __slots__ = "start", "end", "inclusive"

    EMPTY: "Timeframe"
    """Represents an empty timeframe, with a start and end time set to the same value."""

    INFINITE: "Timeframe"
    """Represents an infinite timeframe, with a start time set to the year 1900 and an end time set to the year 2200."""

    def __init__(self, start: datetime, end: datetime, inclusive: bool =False):
        """
        Create a new timeframe. The start- and end-datetime will be stored in the UTC timezone.

        Args:
            start (datetime): Start datetime, this is always inclusive.
            end (datetime): End datetime.
            inclusive (bool): Should the end datetime be inclusive, default is False.
        """
        self.start: datetime = start.astimezone(timezone.utc)
        self.end: datetime = end.astimezone(timezone.utc)
        self.inclusive: bool = inclusive

        assert self.start <= self.end, "start > end"

    @classmethod
    def fromisoformat(cls, start: str, end: str, inclusive : bool =False):
        """
        Create an instance of Timeframe based on a start- and end-datetime in ISO 8601 format.

        Args:
            start (str): Start datetime in ISO 8601 format.
            end (str): End datetime in ISO 8601 format.
            inclusive (bool): Should the end datetime be inclusive, default is False.

        Returns:
            Timeframe: A new Timeframe instance.

        Usage:
            tf1 = Timeframe.fromisoformat("2021-01-01T00:12:00+00:00", "2021-01-02T00:13:00+00:00", True)
            tf2 = Timeframe.fromisoformat("2021-01-01", "2022-01-01", False)
        """
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        return cls(s, e, inclusive)

    def is_empty(self) -> bool:
        """
        Return true if this is an empty timeframe.

        Returns:
            bool: True if the timeframe is empty, False otherwise.
        """
        return self.start == self.end and not self.inclusive

    @staticmethod
    def previous(inclusive : bool = False, days: float=0, seconds: float=0, microseconds: float=0,
                milliseconds: float=0, minutes: float=0, hours: float=0, weeks: float=0) -> "Timeframe":
        """
        Convenient method to create a historic timeframe, the kwargs arguments will be passed to the timedelta.

        Args:
            inclusive (bool): Should the end datetime be inclusive, default is False.
            **kwargs: Arguments to be passed to timedelta.

        Returns:
            Timeframe: A new Timeframe instance.

        Usage:
            tf = Timeframe.previous(days=365)
        """

        td = timedelta(days, seconds, microseconds, milliseconds, minutes, hours, weeks)
        end = datetime.now(timezone.utc)
        start = end - td
        return Timeframe(start, end, inclusive)

    @staticmethod
    def next(inclusive : bool = False,days: float=0, seconds: float=0, microseconds: float=0,
                milliseconds: float=0, minutes: float=0, hours: float=0, weeks: float=0) -> "Timeframe":
        """
        Convenient method to create a future timeframe, the kwargs arguments will be passed to the timedelta.

        Args:
            inclusive (bool): Should the end datetime be inclusive, default is False.
            **kwargs: Arguments to be passed to timedelta.

        Returns:
            Timeframe: A new Timeframe instance.

        Usage:
            tf = Timeframe.next(minutes=30)
        """

        td = timedelta(days, seconds, microseconds, milliseconds, minutes, hours, weeks)
        start = datetime.now(timezone.utc)
        end = start + td
        return Timeframe(start, end, inclusive)

    def __contains__(self, dt: datetime) -> bool:
        """
        Check if a datetime is within the timeframe.

        Args:
            dt (datetime): The datetime to check.

        Returns:
            bool: True if the datetime is within the timeframe, False otherwise.
        """
        if self.inclusive:
            return self.start <= dt <= self.end
        return self.start <= dt < self.end

    def __repr__(self) -> str:
        """
        Return a string representation of the timeframe.

        Returns:
            str: String representation of the timeframe.
        """
        if self.is_empty():
            return "EMPTY_TIMEFRAME"

        last_char = "]" if self.inclusive else ">"
        fmt_str = "%Y-%m-%d %H:%M:%S"
        return f"[{self.start.strftime(fmt_str)} ― {self.end.strftime(fmt_str)}{last_char}"

    @property
    def duration(self) -> timedelta:
        """
        Return the duration of this timeframe expressed as timedelta.

        Returns:
            timedelta: Duration of the timeframe.
        """
        return self.end - self.start

    def annualize(self, rate: float) -> float:
        """
        Annualize the rate for this timeframe.

        Args:
            rate (float): The rate to annualize.

        Returns:
            float: The annualized rate.

        Note:
            At least 1 week is required to calculate annualized return.
        """

        if self.duration < timedelta(weeks=1):
            return float("NaN")

        years = timedelta(days=365) / self.duration
        return (1.0 + rate) ** years - 1.0

    def split(self, n: int | timedelta | Any) -> list["Timeframe"]:
        """
        Split the timeframe in sequential equal parts and return the resulting list of timeframes.

        Args:
            n (int | timedelta | Any): The number of parts or the duration of each part.

        Returns:
            list[Timeframe]: List of resulting timeframes.

        Note:
            The parameter `n` can be a number, a timedelta instance or other types like `relativedelta` that support
            datetime calculations. The last returned timeframe can be shorter than the provided timedelta.
        """

        period = self.duration / n if isinstance(n, int) else n
        end = self.start
        result: list[Timeframe] = []
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
        """
        Sample one or more periods of `duration` with replacements from this timeframe.

        Args:
            duration (timedelta | Any): The duration of each sample.
            n (int): The number of samples to generate, default is 1.

        Returns:
            list[Timeframe]: List of sampled timeframes.

        Raises:
            ValueError: If the sample duration is too large for this timeframe.

        Note:
            It can contain duplicates and the resulting timeframes can overlap.
        """

        result: list[Timeframe] = []
        end = self.end - duration
        if end < self.start:
            raise ValueError("sample duration is too large for this timeframe")

        while len(result) < n:
            start = random.uniform(self.start.timestamp(), end.timestamp())
            start_dt = datetime.fromtimestamp(start, timezone.utc)
            tf = Timeframe(start_dt, start_dt + duration)
            result.append(tf)
        return result

    def __eq__(self, other: Any) -> bool:
        """
        Check if two timeframes are equal.

        Args:
            other (Any): The other timeframe to compare.

        Returns:
            bool: True if the timeframes are equal, False otherwise.
        """
        if isinstance(other, Timeframe):
            return self.start == other.start and self.end == other.end and self.inclusive == other.inclusive

        return False


Timeframe.EMPTY = Timeframe.fromisoformat("1900-01-01T00:00:00+00:00", "1900-01-01T00:00:00+00:00", False)
Timeframe.INFINITE = Timeframe.fromisoformat("1900-01-01T00:00:00+00:00", "2200-01-01T00:00:00+00:00", True)
