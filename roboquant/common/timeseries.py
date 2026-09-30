from collections import UserList
from datetime import datetime, timedelta
from typing import Any

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib import dates, pyplot as plt
from matplotlib.axes import Axes

from roboquant.common.timeframe import Timeframe


class Timeline(UserList[datetime]):
    """A list of sorted datetime objects."""

    @staticmethod
    def from_timeframe(timeframe: Timeframe, step: timedelta | str) -> "Timeline":
        """
        Create a timeline that is within the provided timeframe using step as is increment.
        """

        result = Timeline()

        if isinstance(step, str):
            step = pd.to_timedelta(step)

        time = timeframe.start
        while time in timeframe:
            result.append(time)
            time += step
        return result

    def timeframe(self) -> Timeframe:
        if not self:
            return Timeframe.EMPTY
        return Timeframe(self[0], self[-1], True)


class TimeSeries(pd.DataFrame):
    """A multivariate time-series that contains a timeline and named values.
    Values are always of the type float.

    It is used in several places in roboquant, for example prices and metrics. It contains
    convenience methods to plot the time series.

    Under the hood it extends a Pandas DataFrame with the timeline as an index. So regular
    DataFrame methods also work on TimeSeries objects.
    """

    @property
    def _constructor(self):
        """Override the constructor to return a TimeSeries instead of a plain DataFrame."""
        return TimeSeries

    @staticmethod
    def from_data(timeline: Timeline, data: dict[str, list[float]]) -> "TimeSeries":
        """Create a TimeSeries from a timeline and a dictionary of named data.
        The keys of the dictionary are used as column names and the values are used
        as the data for each column.
        """
        result: TimeSeries = TimeSeries.from_dict(data)
        result.index = timeline
        return result

    @staticmethod
    def univariate(name: str, timeline: Timeline, data: list[float]) -> "TimeSeries":
        """Helper to create a TimeSeries based in single (univariate) dataset"""
        return TimeSeries.from_data(timeline, {name: data})

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        if len(self) == 0:
            return Timeframe.EMPTY

        start: datetime = self.index[0].to_pydatetime(warn=False)
        end: datetime = self.index[-1].to_pydatetime(warn=False)
        return Timeframe(start, end, True)

    def plot_without_timeline(self, *args: Any, **kwargs: Any) -> Axes:
        """Plot the time series without the timeline. This is useful for plotting
        charts when only the values are important and not the absolute timeline.
        """
        return self.reset_index(drop=True).plot(*args, **kwargs)

    def interp(self, timeline: Timeline) -> "TimeSeries":
        """Return a new `TimeSeries` object for the provided timeline using interpolated
        values for the data columns if required.

        Under the hood this method uses the `numpy.interp` function.
        """
        x = [t.timestamp() for t in timeline]
        xp = [t.timestamp() for t in self.timeline()]
        data: dict[str, list[float]] = {}
        for column in self.columns:
            values = np.interp(x, xp, self[column])
            data[column] = values.tolist()
            assert len(values) == len(timeline)
        return TimeSeries.from_data(timeline, data)

    def timeline(self) -> Timeline:
        """Return the timeline of the time series."""
        return Timeline(t.to_pydatetime(warn=False) for t in self.index)

    def limit_timeline(self, timeframe: Timeframe) -> "TimeSeries":
        """Limit the time series of this timeseries to a certain timeframe.
        If the timeframe is empty, an empty time series will be returned."""
        result = self[self.index >= timeframe.start]
        if timeframe.inclusive:
            result = result[result.index <= timeframe.end]
        else:
            result = result[result.index < timeframe.end]
        return result  # type: ignore


    def standardize(self):
        """Standardize the values of the data columns using z-score scaling.

        The used formula is: `(column - mean) / standard_deviation`
        """
        for column in self.columns:
            self[column] = (self[column] - self[column].mean()) / self[column].std()

    def normalize(self):
        """Normalize the values of the data columns using min-max normalization.

        The used formula is: `(column - min) / (max - min)`
        """
        for column in self.columns:
            max = self[column].max()
            min = self[column].min()
            self[column] = (self[column] - min) / (max - min)

    def plot_corr(self, ax: Axes | None = None, plot_colorbar: bool = True, fontsize: int | None = None) -> Axes:
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

    def plot_3d(self, ax: Axes3D | None = None, plot_timeline: bool= True, **kwargs: Any) -> Axes3D:
        """Plot the time series as a 3D area chart.

        Each column is rendered as a separate filled area, stacked along the
        y-axis, with the values plotted along the z-axis.

        Args:
            ax (Axes3D | None): The 3D axes to draw the plot on. If None, a new
                figure and 3D axes are created.
            plot_timeline (bool): Whether to use the timeline (datetime index) for
                the x-axis. If False, the row positions are used instead. Defaults
                to True.
            **kwargs (Any): Additional keyword arguments passed to
                `Axes.fill_between`.

        Returns:
            Axes3D: The 3D axes containing the plot.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.set_yticks([])
        if plot_timeline:
            # matplotlib doesn't support datetime on 3d charts
            # so convert them to a number and set the formatter
            # to auto-date.
            x = dates.date2num(self.index)
            ax.xaxis.set_major_formatter(dates.AutoDateFormatter(dates.AutoDateLocator()))
        else:
            x = range(len(self.index))

        y = 1
        for column in self.columns:
            z = self[column]
            ax.fill_between(x, y, z, x, y, 0, alpha=0.6, label=column, **kwargs)
            y += 1

        plt.legend()
        return ax
