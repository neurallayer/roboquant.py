
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import numpy as np
from numpy.typing import NDArray

import pandas as pd
from matplotlib import pyplot as plt

from roboquant.common.timeframe import Timeframe

Data = list[float] |NDArray[np.float64]
Timeline = list[datetime] | NDArray[np.datetime64]

@dataclass(slots=True)
class TimeSeries:
    """A multivariate time series a timeline an named values. It
    is used in several places in roboquant, for example prices and metrics.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.
    """

    timeline: NDArray[np.datetime64]
    data: dict[str, NDArray[np.float64]]

    def __init__(self, timeline: Timeline, data: dict[str, Data] ):

        # Avoid userwarnings from numpy due to tzinfo
        if len(timeline) and isinstance(timeline[0], datetime):
            timeline = [t.replace(tzinfo=None) for t in timeline]

        self.timeline = np.array(timeline, dtype="datetime64[ms]")

        self.data = { k :np.array(v, dtype=np.float64) for k, v in data.items()}

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

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        if len(self) == 0:
            return Timeframe.EMPTY

        start = self.timeline[0].astype(datetime).replace(tzinfo=timezone.utc)
        end = self.timeline[-1].astype(datetime).replace(tzinfo=timezone.utc)
        return Timeframe(start, end, True)

    def plot(self, plot_timeline: bool = True, ax = None, **kwargs: Any):
        """Plot the data in time series.
        Optional a `matplotlib.axes.Axes` can be provided. If no ax or kwargs are
        provided, some sensible defaults will be used."""
        if ax is None:
            _, ax = plt.subplots()
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if not kwargs:
            kwargs = {"linewidth": 1}

        if plot_timeline:
            for name, data in self.data.items():
                ax.plot(self.timeline, data, label = name, **kwargs)  # type: ignore
        else:
            for name, data in self.data.items():
                ax.plot(self.data, label = name, **kwargs) # type: ignore

        ax.legend()

        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Return the timeseries as a Pandas dataframe with the time being the index
        and the data being the columns.
        """
        df = pd.DataFrame.from_dict(data = self.data) # type: ignore
        df.index = self.timeline
        return df

    def __getitem__(self, key: Any) -> "TimeSeries":
        data: dict[str, Data] = {}
        for name, series in self.data.items():
            data[name] = series.__getitem__(key)

        timeline = self.timeline.__getitem__(key)

        if np.isscalar(timeline):
            for name, series in data.items():
                data[name] = np.array([series])
            timeline = np.array([timeline])
        return TimeSeries(timeline, data)

    def __repr__(self) -> str:
        return f"TimeSeries(series={self.data.keys()} len={len(self)} timeframe={self.timeframe()})"

