# %% [markdown]
# This example shows how to use the Chronos pipeline to predict future prices
# using a pre-trained model.
#
# It uses the `ChronosBoltPipeline` to load the
# pre-trained model and then makes predictions based on historical data.
#
# The predictions are then used in a simple strategy to buy or sell the SPY.

# %%
import torch
from matplotlib import pyplot as plt
import roboquant as rq
import numpy as np
from chronos import ChronosBoltPipeline

from roboquant.strategies.buffer import OHLCVBuffer

# %%
pipeline = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# %%
def perc_change(arr):
    """Calculate the percentage change of a numpy array to make the data stationary."""
    return np.diff(arr) / arr[:-1]


# %%
feed = rq.feeds.YahooFeed("SPY", start_date="2020-01-01")
df = feed.to_dataframe(feed.assets()[0])
close = perc_change(df["Close"].values)
prediction = 10  # predict 10 trading days in the future
context_window = 250  # use the previous 250 trading days as context

# %%
# Get the predictions for different quantiles
result = pipeline.predict(
    context=torch.tensor(close[-context_window - prediction:-prediction]),
    prediction_length=prediction,
)

# %%
# Plot the predictions for each quantile and the actual values
for data in result[0]:
    plt.plot(data.numpy(), color="grey", alpha=0.2)

plt.title("Predictions for SPY")
plt.plot(close[-prediction:], linewidth=2, color="blue")  # type: ignore
plt.show()

# %%
# Make a strategy that uses the predictions to buy or sell the asset
class ChronosStrategy(rq.strategies.TaStrategy):
    """A strategy that uses the Chronos pipeline to predict future prices
    based on historical data. Naive approach that just serves as an example."""

    def __init__(self, pipeline: ChronosBoltPipeline, prediction_length: int):
        super().__init__(period=context_window + 1)
        self.pipeline = pipeline
        self.prediction_length = prediction_length

    def process_asset(self, asset: rq.Asset, ohlcv: OHLCVBuffer) -> rq.Signal | None:
        close = perc_change(ohlcv.close())
        result = self.pipeline.predict(
            context=torch.tensor(close),
            prediction_length=self.prediction_length,
        )

        estimate = result[0, 4].mean().item()  # Get the estimation using the 0.5 quantile

        if estimate > 0.001:
            return rq.Signal.buy(asset)
        elif estimate < -0.001:
            return rq.Signal.sell(asset)


# %%
# Perform a backtest using the strategy
strategy = ChronosStrategy(pipeline, prediction_length=prediction)
trader = rq.traders.FlexTrader(max_order_perc=0.1, max_position_perc=0.8)
account = rq.run(feed, strategy, trader=trader)
print(account)

