
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from roboquant.timeframe import Timeframe

@dataclass(slots=True)
class TimeSeries:
    """A time series contains a name, a timeline and values at each point in time. It
    is used in several places in roboquant, for example prices and metrics.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.
    """

    name: str
    timeline: list[datetime]
    data: list[float]

    def __post_init__(self):
        if len(self.timeline) != len(self.data):
            raise ValueError("Timeline and values must have the same length")

    def __len__(self) -> int:
        return len(self.timeline)

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        return Timeframe(self.timeline[0], self.timeline[-1], True) if len(self) > 0 else Timeframe.EMPTY

    def plot(self, plot_timeline: bool = True, ax = None, **kwargs: Any):
        """Plot the time series. Optional a `matplotlib.axes.Axes` can be provided
        This method requires matplotlib to be installed."""
        if not ax:
            from matplotlib import pyplot as plt
            _, ax = plt.subplots()

        if plot_timeline:
            result = ax.plot(self.timeline, self.data, **kwargs)  # type: ignore
        else:
            result = ax.plot(self.data, **kwargs)

        ax.set_title(self.name)
        return result

    def to_dataframe(self, time_index: bool = False) -> pd.DataFrame:
        """Return the timeseries as a Pandas dataframe optionally with the time being the index
        and the value being the column.
        """

        d = {
            "time": self.timeline,
            self.name: self.data
        }
        df = pd.DataFrame.from_dict(d, orient="columns")
        return df.set_index("time") if time_index else df

    def filter(self, timeframe: Timeframe) -> "TimeSeries":
        """Return a new Timeseries instance which only include observations that fall within the provided timeframe.
        """
        t: list[datetime] = []
        v: list[float] = []
        for idx, time in enumerate(self.timeline):
            if time in timeframe:
                t.append(time)
                v.append(self.data[idx])
        return TimeSeries(self.name, t, v)

