# %%
# uncomment the following line if you didn't install roboquant yet on your environment.
# %pip install --quiet roboquant

# %% [markdown]
# Often you want to visualize some metrics to are specific to your strategy. This example shows how
# to use approach such a case.
# It shows how to use the `IndicatorMetric` and `SignalMetric` to track indicators and signals on a chart.
# It uses the YahooFeed to get the data for TSLA and plots the price, Bollinger Bands, RSI, and buy/sell signals on a chart.
# %%

import roboquant as rq
import matplotlib.pyplot as plt

from roboquant.util.metrics import IndicatorMetric, SignalRatingMetric
from roboquant.util.indicators import BBANDS, RSI

# Setup some defaults for matplotlib
rq.set_dark_style()


# %% [markdown]
# Define the custom metrics we want to use

# %%
class RSIMetric(IndicatorMetric):
    def _calc(self, buffer):
        return {"rsi": RSI(buffer.close, self.timeperiod-1)}

class BBandsMetric(IndicatorMetric):
    def _calc(self, buffer):
        upper, _, lower = BBANDS(buffer.close, timeperiod=self.timeperiod - 1)
        return {"lower": lower, "upper": upper}

# %%
feed = rq.feeds.YahooFeed("TSLA", start_date="2025-01-01", end_date="2026-01-01")
asset = feed.get_asset("TSLA")

# %%
# Subplot ax1 is for the price and BBands and
# subplot ax2 is for the RSI.
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, height_ratios=[4,1,1])


# Plot price chart with bbands
feed.plot(asset, ax = ax1 , plot_volume=False, label="price")
metric = BBandsMetric(asset, timeperiod=10)
bbands = feed.track(metric)
ax1.fill_between(bbands.index, bbands["lower"], bbands["upper"], alpha=0.4, color="grey")  # type: ignore
ax1.set_title(asset.symbol)

# Plot rsi chart
rsi = feed.track(RSIMetric(asset, timeperiod=10))
rsi.plot(ax=ax2)
ax2.axhline(70, color="red", linestyle="--")
ax2.axhline(30, color="green", linestyle="--")
ax2.set_yticks([])
ax2.grid(axis='y')
ax2.legend()

# Plot buy/sell ratings on a chart
strategy = rq.strategies.EMACrossover(2, 5)
metric = SignalRatingMetric(asset, strategy = strategy)
ratings = feed.track(metric)
buy_ratings = ratings[ratings["rating/tsla"] > 0]
sell_ratings = ratings[ratings["rating/tsla"] < 0]
ax3.bar(buy_ratings.index, buy_ratings["rating/tsla"], color="green", label="buy")
ax3.bar(sell_ratings.index, sell_ratings["rating/tsla"], color="red", label="sell")
ax3.legend()

fig.tight_layout(h_pad=0);
# %%
