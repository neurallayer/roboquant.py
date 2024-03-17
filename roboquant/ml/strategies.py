import numpy as np

from roboquant.ml.envs import StrategyEnv, TraderEnv
from roboquant.order import Order
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.traders.trader import Trader


class SB3PolicyStrategy(Strategy):

    def __init__(self, env: StrategyEnv, policy):
        super().__init__()
        self.env = env
        self.policy = policy
        self.state = None

    def create_signals(self, event) -> dict[str, Signal]:
        obs = self.env.get_observation(event)
        if np.any(np.isnan(obs)):
            return {}
        action, self.state = self.policy.predict(obs, state=self.state, deterministic=True)  # type: ignore
        return self.env.get_signals(action, event)

    def reset(self):
        self.state = None


class SB3PolicyTrader(Trader):

    def __init__(self, env: TraderEnv, policy):
        super().__init__()
        self.env = env
        self.policy = policy
        self.state = None

    def create_orders(self, _, event, account) -> list[Order]:
        obs = self.env.get_observation(event, account)
        if np.any(np.isnan(obs)):
            return []
        action, self.state = self.policy.predict(obs, state=self.state, deterministic=True)  # type: ignore
        return self.env.get_orders(action, event, account)

    def reset(self):
        super().reset()
        self.state = None
