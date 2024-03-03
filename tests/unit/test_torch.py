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
        self.lstm = nn.LSTM(6, 8, batch_first=True, num_layers=2, dropout=0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = F.relu(self.flatten(output[:, -1, :]))
        output = self.linear(output)
        return output


class TestTorch(unittest.TestCase):

    def test_lstm_model(self):
        logging.basicConfig()
        logging.getLogger("roboquant.strategies").setLevel(level=logging.INFO)
        # Setup
        symbol = "AAPL"
        prediction = 10
        feed = get_feed()
        model = _MyModel()
        strategy = RNNStrategy(model, symbol, sequences=20, pct=0.01)
        strategy.add_x(CandleFeature(symbol).returns())
        strategy.add_x(SMAFeature(PriceFeature(symbol, "HIGH"), 10).returns())
        strategy.add_y(PriceFeature(symbol, "CLOSE").returns(prediction))

        # Train the model with 10 years of data
        tf = rq.Timeframe.fromisoformat("2010-01-01", "2020-01-01")
        strategy.fit(feed, timeframe=tf, epochs=2, validation_split=0.25, prediction=prediction)

        # Run the trained model with the last 4 years of data
        tf = rq.Timeframe.fromisoformat("2020-01-01", "2023-12-31")
        account = rq.run(feed, strategy, timeframe=tf)
        print(account)
        preds = strategy._prediction_results
        print("avg-prediction=", sum(preds)/len(preds), "   max-prediction=", max(preds))


if __name__ == "__main__":
    unittest.main()
