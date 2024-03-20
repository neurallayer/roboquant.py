import numpy as np

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
