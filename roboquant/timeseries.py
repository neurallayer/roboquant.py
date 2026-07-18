
from dataclasses import dataclass
from datetime import datetime

from roboquant.timeframe import Timeframe


@dataclass
class TimeSeries:
    """A time series contains a name, a timeline and the values of the metric at each point in time. It
    is used in several places in roboquant.

    It contains convenience methods to plot the time series and to convert it to a Pandas dataframe.
    """

    name: str
    timeline: list[datetime]
    values: list[float]

    def __post_init__(self):
        if len(self.timeline) != len(self.values):
            raise ValueError("Timeline and values must have the same length")

    def __len__(self) -> int:
        return len(self.timeline)

    def timeframe(self) -> Timeframe:
        """Return the timeframe of the time series. If the time series is empty,
        an empty timeframe will be returned."""
        return Timeframe(self.timeline[0], self.timeline[-1], True) if len(self) > 0 else Timeframe.EMPTY

    def plot(self, plot_x: bool = True, ax = None, **kwargs):
        """Plot the time series. Optional a `matplotlib.axes.Axes` can be provided
        This method requires matplotlib to be installed."""
        if not ax:
            from matplotlib import pyplot as plt
            _, ax = plt.subplots()

        if plot_x:
            result = ax.plot(self.timeline, self.values, **kwargs)  # type: ignore
        else:
            result = ax.plot(self.values, **kwargs)

        ax.set_title(self.name)
        return result

    def to_dataframe(self, time_index: bool = False):
        """Return the timeseries as a Pandas dataframe optionally with the time being the index
        and the value being the column.
        """
        import pandas as pd

        d = {
            "time": self.timeline,
            self.name: self.values
        }
        df = pd.DataFrame.from_dict(d, orient="columns")
        return df.set_index("time") if time_index else df # type: ignore
