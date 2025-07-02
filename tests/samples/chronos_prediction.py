# %% [markdown]
# This example shows how to use the Chronos pipeline to predict future prices
# using a pre-trained model.
#
# It uses the `ChronosBoltPipeline` to load the
# pre-trained model and then makes predictions based on historical data.
#
# The predictions are then used in a simple strategy to buy or sell the asset.

# %%
import torch
from matplotlib import pyplot as plt
import roboquant as rq
from chronos import BaseChronosPipeline, ChronosBoltPipeline

from roboquant.strategies.buffer import OHLCVBuffer

# %%
pipeline = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# %%
feed = rq.feeds.YahooFeed("SPY", start_date="2020-01-01")
df = feed.to_dataframe(feed.assets()[0])
close = df["Close"].values
prediction = 10  # predict 10 steps in the future
context_window = 250  # use the previous 250 days as context

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
class ChronosPredictionStrategy(rq.strategies.TaStrategy):
    """A strategy that uses the Chronos pipeline to predict future prices
    based on historical data. Naive approach that just serves as an example."""

    def __init__(self, pipeline: BaseChronosPipeline, prediction_length: int):
        super().__init__(period=context_window)
        self.pipeline = pipeline
        self.prediction_length = prediction_length

    def process_asset(self, asset: rq.Asset, ohlcv: OHLCVBuffer) -> rq.Signal | None:
        close = ohlcv.close()
        result = self.pipeline.predict(
            context=torch.tensor(close),
            prediction_length=self.prediction_length,
        )
        last_close = close[-1]
        high_estimate = result[0, 6]  # Get the high estimation using the 0.7 quantile
        low_estimate = result[0, 2]  # Get the low estimation usinf the 0.3 quantile

        if low_estimate.max().item() > last_close:
            return rq.Signal.buy(asset)
        elif high_estimate.min().item() < last_close:
            return rq.Signal.sell(asset)


# %%
# Perform a backtest using the strategy
strategy = ChronosPredictionStrategy(pipeline, prediction_length=prediction)
trader = rq.traders.FlexTrader(max_order_perc=0.1, max_position_perc=0.8)
account = rq.run(feed, strategy, trader=trader)
print(account)

