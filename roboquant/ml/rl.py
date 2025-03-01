import logging
from typing import Callable
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm

import numpy as np
from numpy.typing import NDArray
from roboquant.account import Account

from roboquant.asset import Asset
from roboquant.brokers.simbroker import SimBroker
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.ml.features import Feature
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.timeframe import Timeframe
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.trader import Trader


register(id="roboquant/StrategyEnv-v0", entry_point="roboquant.ml.envs:StrategyEnv")
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """A Gym environment for trading strategies"""
    # pylint: disable=too-many-instance-attributes,unused-argument

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        feed: Feed,
        obs_feature: Feature[Event],
        reward_feature: Feature[Account],
        assets: list[Asset],
        trader: Trader | None = None,
        broker: SimBroker | None = None,
        timeframe: Timeframe | None = None,
        journal_factory: Callable[[str], Journal] | None = None
    ):
        self.broker: SimBroker = broker or SimBroker()
        self.channel = EventChannel()
        self.feed = feed
        self.event: Event | None = None
        self.account: Account = self.broker.sync()
        self.obs_feature = obs_feature
        self.reward_feature = reward_feature
        self.timefame = timeframe
        self.journal_factory = journal_factory
        self.journal: Journal | None = None
        self.epoch = 0
        self.trader = trader or FlexTrader()
        self.assets = assets

        # The observation space is determined by the final shape of the observation feature
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_feature.size(),), dtype=np.float32)

        # The action space is for very asset to predict a number between a strong sell (-1.0) and strogn buy (1.0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.assets),), dtype=np.float32)

        logger.info("observation_space=%s action_space=%s", self.observation_space, self.action_space)

        self.render_mode = None

    def get_observation(self, evt: Event) -> NDArray[np.float32]:
        """Based on an event, calculate the observation features and return them as a Numpy array"""
        return self.obs_feature.calc(evt)

    def get_reward(self, account: Account) -> NDArray[np.float32]:
        """Based on the account, calculate the reward features and return them as a Numpy array"""
        return self.reward_feature.calc(account)

    def step(self, action):
        """Take a step"""
        assert self.event is not None
        assert self.account is not None

        logger.debug("time=%s action=%s", self.event.time, action)

        signals = [Signal(asset, float(rating)) for asset, rating in zip(self.assets, action)]
        orders = self.trader.create_orders(signals, self.event, self.account)
        self.broker.place_orders(orders)

        if self.journal:
            self.journal.track(self.event, self.account, signals, orders)

        self.event = self.channel.get()

        if self.event:
            self.account = self.broker.sync(self.event)

            if self.account.equity_value() < 0.0:
                logger.info("Account equity < 0, done=True")
                return None, 0.0, True, False, {}

            observation = self.get_observation(self.event)
            reward = self.get_reward(self.account)
            return observation, reward, False, False, {}

        return None, 0.0, True, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.broker.reset()
        self.obs_feature.reset()
        self.reward_feature.reset()
        self.epoch += 1

        self.channel = self.feed.play_background(self.timefame)
        if self.journal_factory:
            self.journal = self.journal_factory(f"epoch-{self.epoch}")

        while True:
            self.event = self.channel.get()
            assert self.event is not None, "empty event during warmup"
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event)
            self.get_reward(self.account)
            if not np.any(np.isnan(observation)):
                return observation, {}
            logger.info(observation)

    def render(self):
        """No rendering is supported"""
        pass

    def __repr__(self):
        result = (
            f"TradingEnv(\n\tbroker={self.broker}\n\tfeed={self.feed}"
            f"\n\tfeature_size={self.obs_feature.size()}"
            f"\n\tobservation_space={self.observation_space}\n\taction_space={self.action_space}"
            "\n)"
        )
        return result


class SB3PolicyStrategy(Strategy):
    """A strategy that uses a Stable Baselines 3 policy to generate signals"""

    def __init__(self, obs_feature: Feature[Event], assets: list[Asset], policy: BasePolicy):
        super().__init__()
        self.obs_feature = obs_feature
        self.assets = assets
        self.policy = policy
        self.state = None

    @classmethod
    def from_env(cls, env: TradingEnv, policy: BasePolicy):
        return cls(env.obs_feature, env.assets, policy)

    @classmethod
    def from_model(cls, model: BaseAlgorithm):
        env: TradingEnv = model.env  # type: ignore
        return cls(env.obs_feature, env.assets, model.policy)

    def create_signals(self, event) -> list[Signal]:
        obs = self.obs_feature.calc(event)
        if np.any(np.isnan(obs)):
            return []
        actions, self.state = self.policy.predict(obs, state=self.state, deterministic=True)  # type: ignore
        signals = [Signal(asset, float(rating)) for asset, rating in zip(self.assets, actions)]
        return signals

    def reset(self):
        self.state = None
        self.obs_feature.reset()
