
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from array import array

import pandas as pd
from matplotlib import pyplot as plt

from roboquant.common.timeframe import Timeframe

Data = list[float]
Timeline = list[datetime]

@dataclass(slots=True)
class TimeSeries:
    """A multivariate time series a timeline an named values. It
    is used in several places in roboquant, for example prices and metrics.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.
    """

    timeline: Timeline
    data: dict[str, array[float]]

    def __init__(self, timeline: Timeline, data: dict[str, Data] ):


        self.timeline = timeline

        self.data = { k : array("f", v) for k, v in data.items()}

        for name, value in self.data.items():
            if len(self.timeline) != len(value):
                raise ValueError(f"timeline and {name} have different length")

    @staticmethod
    def univariate(name: str, timeline: Timeline, data: Data) -> "TimeSeries":
        """Helper to create a TimeSeries based in single (univariate) dataset"""
        return TimeSeries(timeline, {name: data})

    def __len__(self) -> int:
        return len(self.timeline)

    @staticmethod
    def empty(name: str):
        return TimeSeries.univariate(name, [], [])

    def names(self):
        return list(self.data.values())

    def append(self, time: datetime, values: dict[str, float]):
        """Add new values to this time-series.

        If a key is missing from the provided values, a NaN will be appended for that missing key.

        Note that this might result in memory reallocations in the underlying Python array,
        so use with cause.
        """
        self.timeline.append(time)
        for k, v in self.data.items():
            new_value = values.get(k, float("nan"))
            v.append(new_value)

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        if len(self) == 0:
            return Timeframe.EMPTY

        start = self.timeline[0]
        end = self.timeline[-1]
        return Timeframe(start, end, True)

    def plot(self, plot_timeline: bool = True, ax = None, **kwargs: Any):
        """Plot the data in time series.
        Optional a `matplotlib.axes.Axes` can be provided. If no ax or kwargs are
        provided, some sensible defaults will be used."""
        if ax is None:
            _, ax = plt.subplots()
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        for name, series in self.data.items():
            _kwargs = {"linewidth": 1, "label" : name} | kwargs
            if plot_timeline:
                ax.plot(self.timeline, series, **_kwargs)  # type: ignore
            else:
                ax.plot(series, **_kwargs)  # type: ignore

        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Return the timeseries as a Pandas dataframe with the time being the index
        and the data being the columns.
        """
        df = pd.DataFrame.from_dict(data = self.data)
        df.index = self.timeline
        return df

    def __getitem__(self, key: Any) -> "TimeSeries":
        data = {}

        for name, series in self.data.items():
            data[name] = series.__getitem__(key)

        timeline = self.timeline.__getitem__(key)

        # Check for single value
        if isinstance(timeline, datetime):
            for name, series in data.items():
                data[name] = [series]
            timeline = [timeline]

        return TimeSeries(timeline, data)

    def __repr__(self) -> str:
        return f"TimeSeries(series={self.data.keys()} len={len(self)} timeframe={self.timeframe()})"

