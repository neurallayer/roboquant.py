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
from roboquant.util.indicators import BBANDS, RSI

# Setup some defaults for matplotlib
rq.set_dark_style()


# %% [markdown]
# Define the custom metrics we want to use

# %%
class RSIMetric(IndicatorMetric):
    def _calc(self, buffer):
        return {"rsi": RSI(buffer.close(), self.timeperiod-1)}

class BBandsMetric(IndicatorMetric):
    def _calc(self, buffer):
        upper, _, lower = BBANDS(buffer.close(), timeperiod=self.timeperiod - 1)
        return {"lower": lower, "upper": upper}


# %%
feed = rq.feeds.YahooFeed("TSLA", start_date="2025-01-01", end_date="2026-01-01")
asset = feed.get_asset("TSLA")

# %%
# Subplot ax1 is for the price and BBands and
# subplot ax2 is for the RSI.
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, height_ratios=[4,1])


# Plot price chart with bbands
feed.plot(asset, ax = ax1 , plot_volume=False, label="price")
metric = BBandsMetric(asset, timeperiod=10)
bbands = feed.track(metric)
ax1.fill_between(bbands.index, bbands["lower"], bbands["upper"], alpha=0.4, color="grey")  # type: ignore
ax1.set_title(asset.symbol);

# Plot rsi chart
rsi = feed.track(RSIMetric(asset, timeperiod=10))
rsi.plot(ax=ax2)
ax2.axhline(70, color="red", linestyle="--", linewidth=0.5)
ax2.axhline(30, color="green", linestyle="--", linewidth=0.5)
ax2.grid(axis='y')
ax2.legend();

fig.tight_layout(h_pad=0);
# %%
