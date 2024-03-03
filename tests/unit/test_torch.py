import logging
import unittest
import roboquant as rq

from roboquant.strategies.features import CandleFeature, PriceFeature, SMAFeature
from roboquant.strategies.torch import RNNStrategy
import torch.nn as nn
import torch.nn.functional as F

from tests.common import get_feed


class _MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(6, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = F.relu(self.flatten(output[:, -1, :]))
        output = F.relu(self.linear1(output))
        output = self.linear2(output)
        return output


class TestRNNStrategy(unittest.TestCase):

    def test_lstm_model(self):
        logging.basicConfig()
        logging.getLogger("roboquant").setLevel(level=logging.INFO)
        # Setup
        symbol = "AAPL"
        feed = get_feed()
        model = _MyModel()
        strategy = RNNStrategy(model, symbol, sequences=20, pct=0.01)
        strategy.add_x(CandleFeature(symbol).returns())
        strategy.add_x(SMAFeature(PriceFeature(symbol, "HIGH"), 10).returns())
        strategy.add_y(PriceFeature(symbol, "CLOSE").returns(10))

        # Train the model with 20 years of data
        tf = rq.Timeframe.fromisoformat("2010-01-01", "2020-01-01")
        strategy.fit(feed, timeframe=tf, epochs=200, validation_split=0.25, prediction=10)

        # Run the trained model with the last 4 years of data
        tf = rq.Timeframe.fromisoformat("2020-01-01", "2023-12-31")
        account = rq.run(feed, strategy, timeframe=tf)
        print(account)
        print(max(strategy._prediction_results))


if __name__ == "__main__":
    unittest.main()
