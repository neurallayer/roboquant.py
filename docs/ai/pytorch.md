---
kernelspec:
  name: python3
  display_name: Python 3
---

# PyTorch

## Overview
Roboquant uses the excellent PyTorch framework as a basis for the included neural-networks based strategies. 
It comes with many popular network architecture layers out of the box and has a Pythonic API, making it
easier to get started. 

## Example
This section provides an example how to develop a strategy that uses PyTorch model to make predictions.

```{code-cell} python
:tags: [remove-input]
import logging
from torch import nn
import torch.nn.functional as F

import roboquant as rq
from roboquant.journals.basicjournal import BasicJournal
from roboquant.ai.features import BarFeature, FeatureSet, MaxReturnFeature, PriceFeature, SMAFeature, VolumeFeature
from roboquant.ai.strategies import TimeSeriesStrategy, logger
```

## Data and Configuration
Here we define some configuration variables, the feed we'll be using,
and the timeframes for training and testing.

```{code-cell} python
prediction_steps = 5 # predict 5 steps in the future
feed = rq.feeds.YahooFeed("AAPL", start_date="2000-01-01")
apple = feed.get_asset("AAPL")
train_tf = rq.Timeframe.fromisoformat("2000-01-01", "2020-01-01")
test_tf = rq.Timeframe.fromisoformat("2020-01-01", "2030-01-01")
```

## Model
We start with defining a LSTM model we want to use. LSTM (Long Short-Term Memory) is a type of
recurrent neural network (RNN) well-suited for  time-series and sequential data.

Unlike standard RNNs, LSTMs can learn long-term dependencies 
by using a gating mechanism that controls the flow of information, making them effective for 
financial time-series prediction where patterns may span many time steps.

Please note this is just an example and likely to overfit with the limited amount of
data we have in this example.

```{code-cell} python
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
```

## Features
Now we define the input and label features we want to use. This is just a small
sample of the available features and custom features can also be added.

```{code-cell} python
input_feature = FeatureSet(
    BarFeature(apple).returns(),
    SMAFeature(PriceFeature(apple), 10).returns(),
    SMAFeature(PriceFeature(apple), 20).returns(),
    SMAFeature(VolumeFeature(apple), 25).returns(),
).normalize(20)

label_feature = MaxReturnFeature(
    PriceFeature(apple, price_type="HIGH"),
    prediction_steps
)
```

## Strategy

Finally we create the actual strategy using the LSTM model we just defined.

```{code-cell} python
model = TimeSeriesLSTM(input_feature.size())
strategy = TimeSeriesStrategy(input_feature, label_feature, model, apple, sequences=20, buy_pct=0.02, sell_pct=0.02)
```

## Training

For fitting we use the timeframe with the first twenty years of data. There is also support for 
validation.

```python
strategy.fit(
    feed,
    timeframe=train_tf,
    epochs=20,
    validation_split=0.25,
    prediction=prediction_steps)
```

## Testing

After the model has been fitted, we can now use the strategy in a test run to see how it performs 
on unseen historic data.

```python
account = rq.run(feed, strategy, timeframe=test_tf)
```

