# %%
# uncomment the following line if you didn't install roboquant yet on your environment.
# %pip install --quiet roboquant

# %% [markdown]
# This example shows how to draw custom metrics on a price plot
# %%
import roboquant as rq
import matplotlib.pyplot as plt

from roboquant.journals.metrics import IndicatorMetric
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
    def __init__(self, asset: rq.Asset):
        super().__init__(asset, 11)

    def _calc(self, buffer):
        upper, _, lower = BBANDS(buffer.close(), timeperiod=10)
        return {"lower": lower, "upper": upper}


# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal(BBandsMetric(asset))
rq.run(feed, strategy, journal=journal)

# %%
ax = feed.plot(asset, plot_volume=False, label="price")
bbands = journal.get_metrics("lower", "upper")
ax.fill_between(bbands.timeline, bbands.data["lower"], bbands.data["upper"], alpha=0.4, color="grey", label="bbands")  # type: ignore
ax.legend()
