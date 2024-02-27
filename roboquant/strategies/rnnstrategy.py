import logging

import torch
from torch.utils.data import Dataset, DataLoader

from roboquant.event import Candle, Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feedutil import play_background
from roboquant.signal import Signal
from roboquant.strategies.buffer import OHLCVBuffer
from roboquant.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class _RNNDataset(Dataset):
    def __init__(self, x_data, y_data, sequences=20):
        self.sequences = sequences
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data) - self.sequences

    def __getitem__(self, idx):
        end = idx + self.sequences
        inputs = self.x_data[idx:end]
        labels = self.y_data[end - 1: end]
        return inputs, labels


class RNNStrategy(Strategy):
    """Use some type of RNN model as a strategy to predict future returns of a single symbol.

    Commonly used models are LSTM models.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            symbol: str,
            capacity: int = 1_000,
            pct: float = 0.01,
            predict_steps: int = 5,
            sequences: int = 20,
            optimizer: torch.optim.Optimizer | None = None,
            criterion: torch.nn.Module | None = None,
    ):
        """
        Args:
        - model: the torch recurrent model to use
        - symbol: the symbol to use
        - sequences: the number of historic steps to use as input to the model
        - future_steps: the number of steps in the future to predict
        - capacity: the maximum capacity for the replay buffer
        - model: the RNN model to use. Input dimensions will be Batch x Sequences x Features and output is Batch x 1
        """
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion or torch.nn.MSELoss()
        self.predict_steps = predict_steps
        self.sequences = sequences
        self.symbol = symbol
        self.pct = pct
        self.ohlcv = OHLCVBuffer(capacity, dtype="float32")

    def pct_change(self, period: int):
        """calculate the percentage change over a certain period"""
        data = self.ohlcv.get_all()
        now = data[period:]
        past = data[:-period]
        return torch.from_numpy(now / past - 1.0)

    def predict_rating(self) -> float | None:
        """Predict based using the data in the replay buffer"""

        if len(self.ohlcv) <= self.sequences:
            return None

        x_data = self.pct_change(1)
        x = x_data[-self.sequences:]
        x = torch.unsqueeze(x, dim=0)  # add the batch dimension
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            p = output.item()
            if p > self.pct:
                return 1.0
            elif p < -self.pct:
                return -1.0
            else:
                return None

    def _train_epoch(self, data_loader):
        model, opt, crit = self.model, self.optimizer, self.criterion

        model.train()
        b, total_loss = 0, torch.tensor([0.0])
        for inputs, labels in data_loader:
            opt.zero_grad()
            output = model(inputs)
            loss = crit(output, labels)
            loss.backward()
            opt.step()
            total_loss += loss.detach()
            b += 1

        return (total_loss / b).item()

    def _valid_epoch(self, data_loader):
        model, crit = self.model, self.criterion
        model.eval()
        b, total_loss = 0, torch.tensor([0.0])
        with torch.no_grad():
            for inputs, labels in data_loader:
                output = model(inputs)
                loss = crit(output, labels)
                total_loss += loss.detach()
                b += 1

        return (total_loss / b).item()

    def _fill_replay_buffer(self, feed, timeframe):
        channel = EventChannel(timeframe)
        play_background(feed, channel)
        h = self.ohlcv
        symbol = self.symbol

        while event := channel.get():
            for item in event.items:
                if isinstance(item, Candle) and item.symbol == symbol:
                    h.append(item.ohlcv)

    def create_signals(self, event: Event) -> dict[str, Signal]:
        item = event.price_items.get(self.symbol)
        if isinstance(item, Candle):
            self.ohlcv.append(item.ohlcv)
            rating = self.predict_rating()
            if rating:
                return {self.symbol: Signal(rating)}
        return {}

    def fit(
            self,
            feed,
            timeframe=None,
            epochs: int = 10,
            batch_size: int = 32,
            validation_split: float = 0.2,
            summary_writer=None,
    ):
        """
        Trains the model for a fixed number of epochs (dataset iterations).
        After the training has completed, this strategy is automatically put in prediction mode.
        """

        self._fill_replay_buffer(feed, timeframe)

        writer = summary_writer

        # make the data stationary
        x_data = self.pct_change(1)
        y_data = self.pct_change(self.predict_steps)[:, 3]

        # what is the boundary between train- and validation-data
        border = round(len(y_data) * (1.0 - validation_split))

        train_dataset = _RNNDataset(x_data[:border], y_data[:border], self.sequences)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataloader = None
        if validation_split > 0.0:
            valid_dataset = _RNNDataset(x_data[border:], y_data[border:], self.sequences)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_dataloader)
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
            logger.info("phase=train epoch=%s/%s loss=%s", epoch + 1, epochs, train_loss)

            if valid_dataloader:
                valid_loss = self._valid_epoch(valid_dataloader)
                if writer:
                    writer.add_scalar("Loss/valid", valid_loss, epoch)
                logger.info("phase=valid epoch=%s/%s loss=%s", epoch + 1, epochs, valid_loss)

        if writer:
            writer.flush()
            writer.close()
