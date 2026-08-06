# %%
import logging
from torch import nn
import torch.nn.functional as F

import roboquant as rq
from roboquant.journals.basicjournal import BasicJournal
from roboquant.ai.features import BarFeature, CombinedFeature, MaxReturnFeature, PriceFeature, SMAFeature, VolumeFeature
from roboquant.ai.strategies import TimeSeriesStrategy, logger


# %%
# PyTorch LSTM Model
class TimeSeriesLSTM(nn.Module):

    def __init__(self, feature_size) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 16, batch_first=True, num_layers=2, dropout=0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = F.relu(self.flatten(output[:, -1, :]))
        output = self.linear(output)
        return output


# %%
# Config
prediction_steps = 5 # predict 5 steps in the future
feed = rq.feeds.YahooFeed("AAPL", start_date="2000-01-01")
apple = feed.get_asset("AAPL")
train_tf = rq.Timeframe.fromisoformat("2000-01-01", "2020-01-01")
test_tf = rq.Timeframe.fromisoformat("2020-01-01", "2030-01-01")

# %%
# Define the strategy

# What are the input features
input_feature = CombinedFeature(
    BarFeature(apple).returns(),
    SMAFeature(PriceFeature(apple), 10).returns(),
    SMAFeature(PriceFeature(apple), 20).returns(),
    SMAFeature(VolumeFeature(apple), 25).returns(),
).normalize(20)

model = TimeSeriesLSTM(input_feature.size())

# What should it predict
# In this case the max return over the prediction period
label_feature = MaxReturnFeature(PriceFeature(apple, price_type="HIGH"), prediction_steps)

# Create the strategy
logging.basicConfig()
logger.setLevel("INFO")
strategy = TimeSeriesStrategy(input_feature, label_feature, model, apple, sequences=20, buy_pct=0.02, sell_pct=0.02)

# %%
# Train the model from 2010 to 20202
strategy.fit(feed, timeframe=train_tf, epochs=20, validation_split=0.25, prediction=prediction_steps)

# %%
# Run the trained model with the last years of data
# logger.setLevel("WARNING")
journal = BasicJournal()
account = rq.run(feed, strategy, timeframe=test_tf, journal=journal)

# %%
# Print some results
print(journal)
print(account)
