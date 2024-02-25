import unittest
from roboquant import Roboquant, Timeframe
from roboquant.traders import FlexTrader
from roboquant.trackers import BasicTracker

from roboquant.strategies.rnnstrategy import RNNStrategy
import torch.nn as nn
import torch.nn.functional as F

from tests.common import get_feed


# Sample model with two LSTM layers followed by a linear layer.
# We keep the model on purpose small since over fitting it is likely
# to happen due to the limited amount of training data.
class _MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(5, 4, batch_first=True, num_layers=2, dropout=0.2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = F.relu(self.flatten(output[:, -1, :]))
        output = self.linear(output)
        return output


class TestRNNStrategy(unittest.TestCase):

    def test_lstm_model(self):
        # Setup
        symbol = "AAPL"
        feed = get_feed()
        model = _MyModel()
        strategy = RNNStrategy(model, symbol, 10_000, 0.001, predict_steps=5, sequences=20)

        # Train the model with 20 years of data
        tf = Timeframe.fromisoformat("2010-01-01", "2019-12-31")
        strategy.fit(feed, tf, epochs=10, validation_split=0.25)

        # Run the trained model with the last 4 years of data
        rq = Roboquant(strategy, FlexTrader(max_order_perc=0.8))
        tf = Timeframe.fromisoformat("2020-01-01", "2023-12-31")
        tracker = BasicTracker()
        rq.run(feed, tracker, tf)
        self.assertGreater(tracker.signals, 0)
        # print(tracker)


if __name__ == "__main__":
    unittest.main()
