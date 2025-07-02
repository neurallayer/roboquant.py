# %%
import logging
import torch
from torch import nn

import roboquant as rq
from roboquant.asset import Stock
from roboquant.journals.basicjournal import BasicJournal
from roboquant.ml.features import BarFeature, CombinedFeature, MaxReturnFeature, PriceFeature, SMAFeature, DayOfMonthFeature
from roboquant.ml.strategies import TimeSeriesStrategy, logger


# %%
# Torch Transformer Model
class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model for time series forecasting, inspired by CodeTrading YouTube channel.
    https://www.youtube.com/watch?v=TT_D-4z-4zY
    """

    def __init__(
        self,
        feature_size,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        label_size=1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, label_size)

    def forward(self, src):
        src = self.input_fc(src)
        src = src + self.pos_embedding

        # Pass through the transformer
        src = self.transformer_encoder(src)

        # We only want the output at the last time step for forecasting the future
        src = src[:, -1, :]

        src = self.fc_out(src)  # [batch_size, prediction_length]
        return src


# %%
# Config
spy = Stock("SPY")
prediction = 5 # predict 5 steps in the future
start_date = "2000-01-01"
feed = rq.feeds.YahooFeed(spy.symbol, start_date=start_date)

# %%

# What are the input features
features = CombinedFeature(
    BarFeature(spy).returns(),
    SMAFeature(PriceFeature(spy), 10).returns(),
    SMAFeature(PriceFeature(spy), 20).returns(),
    DayOfMonthFeature()
).normalize(50)

# What should it predict
# In this case the max return over the prediction period
label = MaxReturnFeature(PriceFeature(spy, price_type="HIGH"), prediction)

model = TimeSeriesTransformer(feature_size=features.size(), seq_length=20, label_size=label.size())

# Create the strategy
logging.basicConfig()
logger.setLevel("INFO")
strategy = TimeSeriesStrategy(features, label, model, spy, sequences=20, buy_pct=0.02, sell_pct=0.02)

# %%
# Train the model from 2000 to 2020
tf = rq.Timeframe.fromisoformat(start_date, "2020-01-01")
strategy.fit(feed, timeframe=tf, epochs=20, validation_split=0.25, prediction=prediction, warmup=100)

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
