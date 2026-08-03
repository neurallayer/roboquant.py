
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from roboquant.common.timeframe import Timeframe

Data = list[float]
Timeline = list[datetime]

class TimeSeries(pd.DataFrame):
    """A multivariate time-series that contains a timeline and named values. It
    is used in several places in roboquant, for example prices and metrics.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.
    """

    @property
    def _constructor(self):
        return TimeSeries

    @staticmethod
    def from_data(timeline: Timeline, data: dict[str, Data]):
        result = TimeSeries.from_dict(data)
        result.index = timeline
        return result

    @staticmethod
    def univariate(name: str, timeline: Timeline, data: Data) -> "TimeSeries":
        """Helper to create a TimeSeries based in single (univariate) dataset"""
        return TimeSeries.from_data(timeline, {name: data})

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        if len(self) == 0:
            return Timeframe.EMPTY

        start = self.index[0]
        end = self.index[-1]
        return Timeframe(start, end, True)

    def plot_me(self, plot_timeline: bool = True, ax = None, **kwargs: Any):
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


    def plot_corr(
        self, ax=None, timeframe: Timeframe | None = None, fontsize : int | None = None
    ):
        """Plot the correlation matrix of various assets in a feed. If no assets are provided,
        all feed assets are used. Returns the main ax.
        """

        if not ax:
            _, ax = plt.subplots()

        corr = self.corr()
        columns = corr.columns

        c_axes = ax.matshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.figure.colorbar(c_axes)

        ax.set_xticks(range(len(columns)), columns, fontsize = fontsize, rotation=45, rotation_mode="xtick")
        ax.set_yticks(range(len(columns)), columns, fontsize = fontsize)

        for (i, j), z in np.ndenumerate(corr.to_numpy()):
            ax.text(
                j,
                i,
                "{:0.2f}".format(z),
                ha="center",
                va="center",
                color="w",
                fontsize = fontsize,
                bbox=dict(boxstyle="round", facecolor='#222', edgecolor='#333', alpha=0.2),
            )

        return ax

    def __repr__(self) -> str:
        return f"TimeSeries(series={self.data.keys()} len={len(self)} timeframe={self.timeframe()})"

