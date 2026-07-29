
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import numpy as np
from numpy.typing import NDArray

import pandas as pd
from matplotlib import pyplot as plt

from roboquant.timeframe import Timeframe

@dataclass(slots=True)
class TimeSeries:
    """A time series contains a name, a timeline and values at each point in time. It
    is used in several places in roboquant, for example prices and metrics.

    Under the hood, all the data is stored in numpy arrays to make further processing faster.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.
    """

    name: str
    timeline: NDArray[np.datetime64]
    data: NDArray[np.float64]

    def __init__(self, name: str, timeline: list[datetime] | NDArray[np.datetime64], data: list[float] |NDArray[np.float64] ):
        self.name = name

        # Avoid userwarnings from numpy due to tzinfo
        if len(timeline) and isinstance(timeline[0], datetime):
            timeline = [t.replace(tzinfo=None) for t in timeline]

        self.timeline = np.array(timeline, dtype="datetime64[ms]")
        self.data = np.array(data, dtype=np.float64)

        if len(self.timeline) != len(self.data):
            raise ValueError("Timeline and data must have the same length")

    def __len__(self) -> int:
        return len(self.timeline)

    @staticmethod
    def empty(name: str):
        return TimeSeries(name, [], [])

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        if len(self) == 0:
            return Timeframe.EMPTY

        start = self.timeline[0].astype(datetime).replace(tzinfo=timezone.utc)
        end = self.timeline[-1].astype(datetime).replace(tzinfo=timezone.utc)
        return Timeframe(start, end, True)

    def plot(self, plot_timeline: bool = True, ax = None, **kwargs: Any):
        """Plot the time series.
        Optional a `matplotlib.axes.Axes` can be provided. If no ax or kwargs are
        provided, some sensible defaults will be used."""
        if ax is None:
            _, ax = plt.subplots()
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if not kwargs:
            kwargs = {"linewidth": 1}

        if plot_timeline:
            ax.plot(self.timeline, self.data, **kwargs)  # type: ignore
        else:
            ax.plot(self.data, **kwargs)

        ax.set_title(self.name)
        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Return the timeseries as a Pandas dataframe with the time being the index
        and the data being the single column.
        """
        return pd.DataFrame(data = self.data, index = self.timeline, columns=[self.name]) # type: ignore

    def filter(self, timeframe: Timeframe) -> "TimeSeries":
        """Return a new Timeseries instance which only include observations that fall within the provided timeframe.
        """
        t: list[datetime] = []
        v: list[float] = []
        for idx, time in enumerate(self.timeline):
            time = time.astype(datetime).replace(tzinfo=timezone.utc)
            if time in timeframe:
                t.append(time)
                v.append(self.data[idx])
        return TimeSeries(self.name, t, v)

    def pct_change(self, name : str | None = None) -> "TimeSeries":
        """Percentage is returned as fraction, so 0.5 is 50%"""
        name = name or self.name
        if len(self) > 1:
            data = np.diff(self.data) / self.data[:-1]
            return TimeSeries(name, self.timeline[1:], data) # type: ignore
        return TimeSeries(name, [], [])

    def inverse_pct_change(self, start: float = 100.0,name : str | None = None) -> "TimeSeries":
        name = name or self.name
        data = np.cumprod(self.data + 1.0) * start
        return TimeSeries(name, self.timeline, data)

    @staticmethod
    def concat(*timeseries: "TimeSeries") -> "TimeSeries":
        """Concatenate multiple TimeSeries together and removed overlap
        between the timeseries. The time-series are expected to be sorted.
        """
        timeline = []
        data = []
        max_time: np.datetime64 = np.datetime64("1900")
        for ts in timeseries:
            entries = ts.timeline > max_time
            timeline.extend(ts.timeline[entries])
            data.extend(ts.data[entries])
            if timeline:
                max_time = timeline[-1]

        return TimeSeries(timeseries[0].name, timeline, data)

    def __mul__(self, other: float) -> "TimeSeries":
        return TimeSeries(self.name, self.timeline, self.data * other)

    def __add__(self, other: float) -> "TimeSeries":
        return TimeSeries(self.name, self.timeline, self.data + other)


