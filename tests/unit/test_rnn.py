import unittest

from torch import nn
import torch.nn.functional as F
import numpy as np

import roboquant as rq
from roboquant.asset import Stock
from roboquant.ml.features import BarFeature, CombinedFeature, PriceFeature, SMAFeature
from roboquant.ml.strategies import RNNStrategy, SequenceDataset
from tests.common import get_feed


class _MyModel(nn.Module):
    """Sample LSTM based model for testign purposes"""

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

    def test_dataset(self):
        x = np.ones((100, 10))
        y = np.ones((100, 5))
        ds = SequenceDataset(x, y, 20, 10, 1)
        size = len(ds)
        self.assertEqual(70, size)
        for idx in range(size):
            a, b = ds[idx]
            self.assertEqual(20, len(a))
            self.assertEqual(10, len(b))

    def test_lstm_model(self):
        # logging.basicConfig()
        # logging.getLogger("roboquant.strategies").setLevel(level=logging.INFO)
        # Setup
        apple = Stock("AAPL")
        prediction = 5
        feed = get_feed()
        model = _MyModel()

        input_feature = CombinedFeature(
            BarFeature(apple),
            SMAFeature(PriceFeature(apple, price_type="HIGH"), 10)
        ).returns().normalize()

        label_feature = PriceFeature(apple, price_type="CLOSE").returns(prediction)

        strategy = RNNStrategy(input_feature, label_feature, model, apple, sequences=20, buy_pct=0.01)

        # Train the model with 10 years of data
        tf = rq.Timeframe.fromisoformat("2010-01-01", "2020-01-01")
        strategy.fit(feed, timeframe=tf, epochs=2, validation_split=0.50, prediction=prediction)

        # Run the trained model with the last 4 years of data
        tf = rq.Timeframe.fromisoformat("2020-01-01", "2024-01-01")
        account = None
        try:
            account = rq.run(feed, strategy, timeframe=tf)
        except:  # noqa: E722
            pass
        self.assertTrue(account)


if __name__ == "__main__":
    unittest.main()
