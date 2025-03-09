# %%
import logging
from torch import nn
import torch.nn.functional as F

import roboquant as rq
from roboquant.asset import Stock
from roboquant.journals.basicjournal import BasicJournal
from roboquant.ml.features import BarFeature, CombinedFeature, MaxReturnFeature, PriceFeature, SMAFeature
from roboquant.ml.strategies import RNNStrategy, logger


# %%
# Torch LSTM Model
class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(15, 16, batch_first=True, num_layers=2, dropout=0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = F.relu(self.flatten(output[:, -1, :]))
        output = self.linear(output)
        return output


# %%
# Config
apple = Stock("AAPL")
prediction = 5 # predict 5 steps in the future
start_date = "2010-01-01"
feed = rq.feeds.YahooFeed(apple.symbol, start_date=start_date)

# %%
# Define the strategy
model = MyModel()

# What are the input features
input_feature = CombinedFeature(
    BarFeature(apple).returns(),
    SMAFeature(BarFeature(apple), 10).returns(),
    SMAFeature(BarFeature(apple), 20).returns(),
).normalize(20)

# What should it predict
# In this case the max return over the prediction period
label_feature = MaxReturnFeature(PriceFeature(apple, price_type="HIGH"), prediction)

# Create the strategy
logging.basicConfig()
logger.setLevel("INFO")
strategy = RNNStrategy(input_feature, label_feature, model, apple, sequences=20, buy_pct=0.02, sell_pct=0.02)

# %%
# Train the model from 2010 to 20202
tf = rq.Timeframe.fromisoformat(start_date, "2020-01-01")
strategy.fit(feed, timeframe=tf, epochs=20, validation_split=0.25, prediction=prediction)

# %%
# Run the trained model with the last years of data
logger.setLevel("WARNING")
tf = rq.Timeframe.fromisoformat("2020-01-01", "2025-01-01")
journal = BasicJournal()
account = rq.run(feed, strategy, timeframe=tf, journal=journal)

# %%
# Print some results
print(journal)
print(account)
