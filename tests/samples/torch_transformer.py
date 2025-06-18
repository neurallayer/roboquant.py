# %%
import logging
import torch
from torch import nn

import roboquant as rq
from roboquant.asset import Stock
from roboquant.journals.basicjournal import BasicJournal
from roboquant.ml.features import BarFeature, CombinedFeature, MaxReturnFeature, PriceFeature, SMAFeature
from roboquant.ml.strategies import TimeSeriesStrategy, logger


# %%
# Torch Transformer Model
class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model for time series forecasting inspired by CodeTrading YouTube channel."""

    def __init__(
        self,
        feature_size,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        prediction_length=1
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
            activation="relu",
            batch_first=True  # Ensure batch is first dimension
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, src):

        # src shape: [batch_size, seq_length, feature_size]
        seq_len = src.shape[1]
        src = self.input_fc(src)  # -> [batch_size, seq_length, d_model]
        src = src + self.pos_embedding[:, :seq_len, :]

        # Transformer expects shape: [sequence_length, batch_size, d_model]
        src = src.permute(1, 0, 2)  # -> [seq_length, batch_size, d_model]

        # Pass through the transformer
        encoded = self.transformer_encoder(src)  # [seq_length, batch_size, d_model]

        # We only want the output at the last time step for forecasting the future
        last_step = encoded[-1, :, :]  # [batch_size, d_model]

        out = self.fc_out(last_step)  # [batch_size, prediction_length]
        return out


# %%
# Config
apple = Stock("AAPL")
prediction = 5 # predict 5 steps in the future
start_date = "2000-01-01"
feed = rq.feeds.YahooFeed(apple.symbol, start_date=start_date)

# %%

# What are the input features
input_feature = CombinedFeature(
    BarFeature(apple).returns(),
    SMAFeature(BarFeature(apple), 10).returns(),
    SMAFeature(BarFeature(apple), 20).returns(),
).normalize(20)

model = TimeSeriesTransformer(feature_size=input_feature.size(), seq_length=20)

# What should it predict
# In this case the max return over the prediction period
label_feature = MaxReturnFeature(PriceFeature(apple, price_type="HIGH"), prediction)

# Create the strategy
logging.basicConfig()
logger.setLevel("INFO")
strategy = TimeSeriesStrategy(input_feature, label_feature, model, apple, sequences=20, buy_pct=0.02, sell_pct=0.02)

# %%
# Train the model from 2000 to 2020
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
