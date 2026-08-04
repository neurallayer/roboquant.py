from datetime import datetime

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

    Under the hood it is a Pandas DataFrame with the timeline as an index. So regular DataFrame
    methods also work on TimeSeries objects.
    """

    @property
    def _constructor(self):
        return TimeSeries

    @staticmethod
    def from_data(timeline: Timeline, data: dict[str, Data]) -> "TimeSeries":
        result : TimeSeries = TimeSeries.from_dict(data) # type: ignore
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

        start = self.index[0].to_pydatetime(warn=False) # type: ignore
        end = self.index[-1].to_pydatetime(warn=False) # type: ignore
        return Timeframe(start, end, True)

    def plot_corr(self, ax=None, timeframe: Timeframe | None = None, fontsize: int | None = None):
        """Plot the correlation matrix of the series."""

        if not ax:
            _, ax = plt.subplots()

        corr = self.corr()
        columns = corr.columns

        c_axes = ax.matshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.figure.colorbar(c_axes)

        ax.set_xticks(range(len(columns)), columns, fontsize=fontsize, rotation=45, rotation_mode="xtick")
        ax.set_yticks(range(len(columns)), columns, fontsize=fontsize)

        for (i, j), z in np.ndenumerate(corr.to_numpy()):
            ax.text(
                j,
                i,
                "{:0.2f}".format(z),
                ha="center",
                va="center",
                color="w",
                fontsize=fontsize,
                bbox=dict(boxstyle="round", facecolor="#222", edgecolor="#333", alpha=0.2),
            )

        return ax
