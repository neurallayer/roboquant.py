from datetime import datetime
from typing import Any

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from roboquant.common.timeframe import Timeframe

Data = list[float]
Timeline = list[datetime]


class TimeSeries(pd.DataFrame):
    """A multivariate time-series that contains a timeline and named values.
    Values are always of the type float.

    It is used in several places in roboquant, for example prices and metrics.

    It contains convenience methods to plot the time series or to convert it to a Pandas dataframe.

    Under the hood it is a Pandas DataFrame with the timeline as an index. So regular DataFrame
    methods also work on TimeSeries objects.
    """

    @property
    def _constructor(self):
        """Override the constructor to return a TimeSeries instead of a plain DataFrame."""
        return TimeSeries

    @staticmethod
    def from_data(timeline: Timeline, data: dict[str, Data]) -> "TimeSeries":
        """Create a TimeSeries from a timeline and a dictionary of named data.
        The keys of the dictionary are used as column names and the values are used
        as the data for each column."""
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

    def plot_without_timeline(self, *args: Any, **kwargs: Any) -> Axes:
        """Plot the time series without the timeline. This is useful for plotting
        charts when only the values are important and not the absolute timeline.
        """
        return self.reset_index(drop=True).plot(*args, **kwargs)

    def timeline(self) -> list[datetime]:
        """Return the timeline of the time series as a list of datetime objects."""
        return [t.to_pydatetime(warn=False) for t in self.index]

    def limit_timeline(self, timeframe: Timeframe) -> "TimeSeries":
        """Limit the time series to a certain timeframe. If the timeframe is empty,
        an empty time series will be returned."""
        result = self[self.index >= timeframe.start]
        if timeframe.inclusive:
            result = result[result.index <= timeframe.end]
        else:
            result = result[result.index < timeframe.end]
        return result # type: ignore

    def plot_corr(self, ax : Axes | None =None, plot_colorbar: bool = True, fontsize: int | None = None) -> Axes:
        """Plot the correlation matrix of the series."""

        if not ax:
            _, ax = plt.subplots()

        corr = self.corr()
        columns = corr.columns

        c_axes = ax.matshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
        if plot_colorbar:
            ax.figure.colorbar(c_axes)

        ax.grid(False)
        ax.set_xticks(range(len(columns)), columns, fontsize=fontsize, rotation=45)
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
