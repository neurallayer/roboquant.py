import logging
import unittest

from torch import nn
import torch.nn.functional as F

import roboquant as rq
from roboquant.journals.basicjournal import BasicJournal
from roboquant.strategies.features import CandleFeature, MaxReturnFeature, PriceFeature, SMAFeature
from roboquant.strategies.torch import RNNStrategy


class _MyModel(nn.Module):
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


class TestTorch(unittest.TestCase):

    def test_lstm_model(self):
        logging.basicConfig()
        logging.getLogger("roboquant.strategies").setLevel(level=logging.INFO)

        # Config
        symbol = "AAPL"
        prediction = 10
        start_date = "2010-01-01"
        feed = rq.feeds.YahooFeed(symbol, start_date=start_date)

        # Define the stategy
        model = _MyModel()
        strategy = RNNStrategy(model, symbol, sequences=20, buy_pct=0.04, sell_pct=0.01)
        strategy.add_x(CandleFeature(symbol).returns())
        strategy.add_x(SMAFeature(CandleFeature(symbol), 10).returns())
        strategy.add_x(SMAFeature(CandleFeature(symbol), 20).returns())
        strategy.add_y(MaxReturnFeature(PriceFeature(symbol, "HIGH"), prediction))

        # Train the model
        tf = rq.Timeframe.fromisoformat(start_date, "2020-01-01")
        strategy.fit(feed, timeframe=tf, epochs=20, validation_split=0.25, prediction=prediction)

        # Run the trained model with the last 4 years of data
        tf = rq.Timeframe.fromisoformat("2020-01-01", "2024-01-01")
        journal = BasicJournal()
        account = rq.run(feed, strategy, timeframe=tf, journal=journal)

        # Print some results
        print(journal)
        predictions = strategy.prediction_results
        print(max(predictions), min(predictions))
        print(account)


if __name__ == "__main__":
    unittest.main()
