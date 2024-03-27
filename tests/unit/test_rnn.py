import unittest

from torch import nn
import torch.nn.functional as F

import roboquant as rq
from roboquant.ml.features import BarFeature, CombinedFeature, PriceFeature, SMAFeature
from roboquant.ml.strategies import RNNStrategy
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


class TestRNN(unittest.TestCase):

    def test_lstm_model(self):
        # logging.basicConfig()
        # logging.getLogger("roboquant.strategies").setLevel(level=logging.INFO)
        # Setup
        symbol = "AAPL"
        prediction = 10
        feed = get_feed()
        model = _MyModel()

        input_feature = CombinedFeature(
            BarFeature(symbol),
            SMAFeature(PriceFeature(symbol, price_type="HIGH"), 10)
        ).returns().normalize()

        label_feature = PriceFeature(symbol, price_type="CLOSE").returns(prediction)

        strategy = RNNStrategy(input_feature, label_feature, model, symbol, sequences=20, buy_pct=0.01)

        # Train the model with 10 years of data
        tf = rq.Timeframe.fromisoformat("2010-01-01", "2020-01-01")
        strategy.fit(feed, timeframe=tf, epochs=2, validation_split=0.25, prediction=prediction)

        # Run the trained model with the last 4 years of data
        tf = rq.Timeframe.fromisoformat("2020-01-01", "2024-01-01")
        rq.run(feed, strategy, timeframe=tf)
        predictions = strategy.prediction_results
        self.assertEqual(987, len(predictions))
        self.assertNotEqual(max(predictions), min(predictions))


if __name__ == "__main__":
    unittest.main()
