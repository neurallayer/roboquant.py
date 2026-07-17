# %% [markdown]
# Chronos is a time series forecasting library that can be used to predict future values
# based on historical data. It is built on top of PyTorch and provides a simple interface
# for training and using models for time series forecasting.
#
# This example uses the `ChronosBoltPipeline` to load the pre-trained model and then makes
# predictions based on historical data.
#

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
prediction_days = 10  # predict 10 trading days in the future
context_window = 250  # use the previous 250 trading days as context

feed = rq.feeds.YahooFeed("SPY", start_date="2015-01-01")
df = feed.to_dataframe(feed.assets()[0])
close = df["Close"].values
close = perc_change(close)  # make the data stationary
close = close[-context_window - prediction_days :-prediction_days]  # use the last 250 trading days as context

# %%
# Get the predictions for different quantiles based on last 250 trading days of data
quantiles = pipeline.predict(
    inputs=torch.tensor(close),
    prediction_length=prediction_days,
)

# %%
# Plot the predictions for each quantile and the actual values
plt.title(f"Predictions for SPY changes in the next {prediction_days} trading days")

for quantile in quantiles[0]:
    plt.plot(quantile.numpy(), color="grey", alpha=0.5)

plt.plot(close[-prediction_days:], linewidth=1, color="blue")  # type: ignore
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
            inputs=torch.tensor(close),
            prediction_length=self.prediction_length,
        )

        # Get the estimations using the 0.5 quantile
        # shape is (batch_size, num_quantiles, prediction_length)
        mean = result[0, 4].mean().item()
        high = result[0, 4].max().item()
        low = result[0, 4].min().item()

        if mean > 0 and high > 0.001:
            return rq.Signal.buy(asset)
        elif mean < 0 and low < -0.001:
            return rq.Signal.sell(asset)


# %%
# Perform a backtest using the strategy
strategy = ChronosStrategy(pipeline, prediction_length=prediction_days)
trader = rq.traders.FlexTrader(max_order_perc=0.1, max_position_perc=0.8)
account = rq.run(feed, strategy, trader=trader)
print(account)

