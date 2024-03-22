from abc import abstractmethod
from collections import deque
import numpy as np
from numpy.typing import NDArray

from roboquant.event import Event
from roboquant.ml.envs import Action2Signals, StrategyEnv, TraderEnv
from roboquant.ml.features import Feature
from roboquant.order import Order
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.traders.trader import Trader


class SB3PolicyStrategy(Strategy):

    def __init__(self, obs_feature: Feature, action_2_signals: Action2Signals, policy):
        super().__init__()
        self.obs_feature = obs_feature
        self.action_2_signals = action_2_signals
        self.policy = policy
        self.state = None

    @classmethod
    def from_env(cls, env: StrategyEnv, policy):
        return cls(env.obs_feature, env.action_2_signals, policy)

    def create_signals(self, event) -> dict[str, Signal]:
        obs = self.obs_feature.calc(event, None)
        if np.any(np.isnan(obs)):
            return {}
        action, self.state = self.policy.predict(obs, state=self.state, deterministic=True)  # type: ignore
        return self.action_2_signals.get_signals(action, event)

    def reset(self):
        self.state = None
        self.obs_feature.reset()


class SB3PolicyTrader(Trader):

    def __init__(self, obs_feature: Feature, action_2_orders, policy):
        super().__init__()
        self.obs_feature = obs_feature
        self.action_2_orders = action_2_orders
        self.policy = policy
        self.state = None

    @classmethod
    def from_env(cls, env: TraderEnv, policy):
        return cls(env.obs_feature, env.action_2_orders, policy)

    def create_orders(self, _, event, account) -> list[Order]:
        obs = self.obs_feature.calc(event, account)
        if np.any(np.isnan(obs)):
            return []
        action, self.state = self.policy.predict(obs, state=self.state, deterministic=True)  # type: ignore
        return self.action_2_orders.get_orders(action, event, account)

    def reset(self):
        super().reset()
        self.state = None
        self.obs_feature.reset()


class FeatureStrategy(Strategy):
    """Abstract base class for strategies wanting to use features
    for their input and target.
    """

    def __init__(self, input_feature: Feature, label_feature: Feature, history: int, dtype="float32"):
        self._features_x = []
        self._features_y = []
        self.input_feature = input_feature
        self.label_feature = label_feature
        self._hist = deque(maxlen=history)
        self._dtype = dtype

    def create_signals(self, event: Event) -> dict[str, Signal]:
        h = self._hist
        row = self.input_feature.calc(event, None)
        h.append(row)
        if len(h) == h.maxlen:
            x = np.asarray(h, dtype=self._dtype)
            return self.predict(x)
        return {}

    @abstractmethod
    def predict(self, x: NDArray) -> dict[str, Signal]: ...

    def _get_xy(self, feed, timeframe=None, warmup=0) -> tuple[NDArray, NDArray]:
        channel = feed.play_background(timeframe)
        x = []
        y = []
        while evt := channel.get():
            if warmup:
                self.label_feature.calc(evt, None)
                self.input_feature.calc(evt, None)
                warmup -= 1
            else:
                x.append(self.input_feature.calc(evt, None))
                y.append(self.label_feature.calc(evt, None))

        return np.asarray(x, dtype=self._dtype), np.asarray(y, dtype=self._dtype)
