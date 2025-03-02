import logging
from abc import abstractmethod
from collections import deque
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray

from torch.utils.data import DataLoader, Dataset

from roboquant.asset import Asset
from roboquant.event import Event
from roboquant.ml.features import Feature, NormalizeFeature
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class FeatureStrategy(Strategy):
    """Abstract base class for strategies wanting to use event features
    for their input.
    """

    def __init__(self, input_feature: Feature[Event], history: int, dtype="float32"):
        super().__init__()
        self.input_feature = input_feature
        self.history = history
        self._hist = deque(maxlen=history)
        self._dtype = dtype

    def create_signals(self, event: Event) -> list[Signal]:
        h = self._hist
        row = self.input_feature.calc(event)
        h.append(row)
        if len(h) == h.maxlen:
            x = np.asarray(h, dtype=self._dtype)
            return self.predict(x, event.time)
        return []

    @abstractmethod
    def predict(self, x: NDArray, dt: datetime) -> list[Signal]:
        """Subclasses need to implement this method"""
        ...


class SequenceDataset(Dataset):
    """Dataset that creates an input sequence and an output sequence useful for recurrent networks.
    The output sequence is always after the input sequence, but there can be
    optionally a gap.

    ```
    [...input...][...gap...][...target...]
    ```
    """

    def __init__(
        self,
        input_data: NDArray,
        target_data: NDArray,
        input_sequences=20,
        target_sequences=1,
        gap=0,
        transform=None,
        target_transform=None,
        target_squeeze=True
    ):
        assert len(input_data) == len(target_data), "x_data and y_data need to have the same length"
        self.input_data = input_data
        self.target_data = target_data
        self.input_sequences = input_sequences
        self.output_sequences = target_sequences
        self.gap = gap
        self.transform = transform
        self.target_squeeze = target_squeeze
        self.target_transform = target_transform

        if len(self) == 0:
            logger.warning("this dataset won't produce any data")

    def __len__(self):
        calc_l = len(self.target_data) - self.input_sequences - self.output_sequences - self.gap + 1
        return max(0, calc_l)

    def __getitem__(self, idx):
        end = idx + self.input_sequences
        features = self.input_data[idx:end]
        start = end + self.gap
        target = self.target_data[start: start + self.output_sequences]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)

        if self.output_sequences == 1 and self.target_squeeze:
            target = np.squeeze(target, 0)
        return features, target


class RNNStrategy(FeatureStrategy):
    """A strategy that uses a recurrent neural network to predict the future of a time series.
    The input and label features are both features that can be calculated from an event."""

    def __init__(
        self,
        input_feature: Feature[Event],
        label_feature: Feature[Event],
        model: torch.nn.Module,
        asset: Asset,
        sequences: int = 20,
        buy_pct: float = 0.01,
        sell_pct=0.0,
    ):
        super().__init__(input_feature, sequences)
        self.label_feature = label_feature
        self.model = model
        self.buy_pct = buy_pct
        self.sell_pct = sell_pct
        self.asset = asset

    def predict(self, x, dt) -> list[Signal]:
        x = torch.asarray(x)
        x = torch.unsqueeze(x, dim=0)  # add the batch dimension

        self.model.eval()
        with torch.no_grad():
            output = self.model(x).numpy()

            if isinstance(self.label_feature, NormalizeFeature):
                p = self.label_feature.denormalize(output).item()
            else:
                p = output.item()

            logger.info("prediction p=%s time=%s", p, dt)
            if p >= self.buy_pct:
                return [Signal.buy(self.asset)]
            if p <= self.sell_pct:
                return [Signal.sell(self.asset)]
        return []

    def _get_dataloaders(self, x, y, prediction: int, validation_split: float, batch_size: int):
        # what is the border between train- and validation-data
        border = round(len(y) * (1.0 - validation_split))

        x_train = x[:border]
        y_train = y[:border]

        train_dataset = SequenceDataset(x_train, y_train, self.history, gap=prediction)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataloader = None
        if validation_split > 0.0:
            x_valid = x[border:]
            y_valid = y[border:]
            valid_dataset = SequenceDataset(x_valid, y_valid, self.history, gap=prediction)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, valid_dataloader

    def __get_xy(self, feed, timeframe=None, warmup=0) -> tuple[NDArray, NDArray]:
        channel = feed.play_background(timeframe)
        x = []
        y = []
        while evt := channel.get():
            if warmup:
                self.label_feature.calc(evt)
                self.input_feature.calc(evt)
                warmup -= 1
            else:
                x.append(self.input_feature.calc(evt))
                y.append(self.label_feature.calc(evt))

        return np.asarray(x, dtype=self._dtype), np.asarray(y, dtype=self._dtype)

    @staticmethod
    def describe(x):
        print("shape=", x.shape, "min=", np.min(x, axis=0), "max=", np.max(x, axis=0), "mean=", np.mean(x, axis=0))

    def fit(
        self,
        feed,
        optimizer=None,
        criterion=None,
        prediction=1,
        timeframe=None,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        warmup=50,
        writer=None,
    ):
        """
        Train the model for a fixed number of epochs (dataset iterations).

        Args:
            feed: The data feed to use.
            optimizer: The torch optimizer to use.
            If None is specified, Adam will be used.
            criterion: The torch loss function to use. If None is specified, MESLoss will be used.
            prediction: The steps in the future to predict, default is 1.
            timeframe: The timeframe to limit the training and validation to.
            epochs: The total number of epochs to train the model, default is 10.
            batch_size: The batch size to use, default is 32.
            validation_split: the percentage to use for validation, default is 0.20 (20%).
            warmup: nuber of warmup steps
            writer: the tensorboard writer to use to log losses, default is None.
        """
        optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = criterion or torch.nn.MSELoss()

        x, y = self.__get_xy(feed, timeframe, warmup=warmup)
        logger.info("x-shape=%s", x.shape)
        logger.info("y-shape=%s", y.shape)

        train_dataloader, valid_dataloader = self._get_dataloaders(x, y, prediction, validation_split, batch_size)

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_dataloader, optimizer, criterion)
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
            logger.info("phase=train epoch=%s/%s loss=%s", epoch + 1, epochs, train_loss)

            if valid_dataloader:
                valid_loss = self._valid_epoch(valid_dataloader, criterion)
                if writer:
                    writer.add_scalar("Loss/valid", valid_loss, epoch)
                logger.info("phase=valid epoch=%s/%s loss=%s", epoch + 1, epochs, valid_loss)

        if writer:
            writer.flush()

    def _train_epoch(self, data_loader, opt, crit):
        model = self.model
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

    def _valid_epoch(self, data_loader, crit):
        model = self.model
        model.eval()
        b, total_loss = 0, torch.tensor([0.0])
        with torch.no_grad():
            for inputs, labels in data_loader:
                output = model(inputs)
                loss = crit(output, labels)
                total_loss += loss.detach()
                b += 1

        return (total_loss / b).item()
