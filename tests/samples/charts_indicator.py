# %%
# uncomment the following line if you didn't install roboquant yet on your environment.
# %pip install --quiet roboquant

# %% [markdown]
# This example shows how to draw custom metrics on a price plot. It uses the Bollinger Bands indicator
# as an example. The Bollinger Bands are calculated using the `BBANDS` function from the
# `roboquant.util.indicators` module.
#
# The `BBandsMetric` class extends the `IndicatorMetric` class and implements
# the `_calc` method to calculate the upper and lower bands based on the closing
# prices of the asset. The results are then plotted alongside the price data
# using Matplotlib.
# %%
import roboquant as rq
import matplotlib.pyplot as plt

from roboquant.util.metrics import IndicatorMetric
from roboquant.util.indicators import BBANDS

# Setup some defaults for matplotlib
plt.style.use("dark_background")
plt.rcParams["figure.figsize"] = [10.0, 5.0]
plt.rcParams["figure.dpi"] = 150

# %%
feed = rq.feeds.YahooFeed("TSLA", start_date="2025-01-01", end_date="2026-01-01")
asset = feed.get_asset("TSLA")


# %%
class BBandsMetric(IndicatorMetric):
    def _calc(self, buffer):
        upper, _, lower = BBANDS(buffer.close(), timeperiod=self.timeperiod - 1)
        return {"lower": lower, "upper": upper}


# %%
ax = feed.plot(asset, plot_volume=False, label="price")
metric = BBandsMetric(asset, timeperiod=10)
bbands = feed.track(metric)
ax.fill_between(bbands.timeline, bbands.data["lower"], bbands.data["upper"], alpha=0.4, color="grey", label="bbands")  # type: ignore
ax.legend();
