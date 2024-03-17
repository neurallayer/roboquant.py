from decimal import Decimal
import logging
from typing import Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from roboquant.account import Account

from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.order import Order
from roboquant.signal import Signal
from roboquant.ml.features import Feature
from roboquant.ml.torch import Normalize
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.trader import Trader


logger = logging.getLogger(__name__)


class StrategyEnv(gym.Env):
    # pylint: disable=too-many-instance-attributes,unused-argument

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        features: Sequence[Feature],
        feed: Feed,
        rating_symbols: list[str],
        broker: Broker | None = None,
        trader: Trader | None = None,
    ):
        self.broker: Broker = broker or SimBroker()
        self.trader: Trader = trader or FlexTrader()
        self.channel = EventChannel()
        self.feed = feed
        self.event: Event | None = None
        self.account: Account = self.broker.sync()
        self.symbols = rating_symbols
        self.features = features
        self.last_equity: float = self.account.equity()
        self.obs_normalizer = None
        self.reward_normalizer = None
        self.enable_cache = True
        self._cache = {}

        action_size = len(rating_symbols)
        obs_size = sum(feature.size() for feature in features)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_size,), dtype=np.float32)

        logger.info("observation_space=%s action_space=%s", self.observation_space, self.action_space)

        self.render_mode = None

    def calc_normalization(self, steps: int):
        enable_cache = self.enable_cache
        self.enable_cache = False
        obs, _ = self.reset()
        obs_buffer = np.zeros((steps, obs.shape[0]), dtype=np.float32)
        reward_buffer = np.zeros((steps,), dtype=np.float32)
        step = 0
        while step < steps:
            action = self.action_space.sample()
            obs, reward, terminated, _, _ = self.step(action)
            if not terminated:
                obs_buffer[step] = obs
                reward_buffer[step] = reward
            else:
                self.reset()
            step += 1

        obs_norm = obs_buffer.mean(axis=0), obs_buffer.std(axis=0)
        reward_norm = reward_buffer.mean(axis=0).item(), reward_buffer.std(axis=0).item()
        self.obs_normalizer = Normalize(obs_norm)
        self.reward_normalizer = Normalize(reward_norm)
        self.enable_cache = enable_cache

    def get_observation(self, evt: Event) -> NDArray[np.float32]:
        if self.enable_cache and evt.time in self._cache:
            return self._cache[evt.time]
        data = [feature.calc(evt, None) for feature in self.features]
        obs = np.hstack(data, dtype=np.float32)
        result = self.obs_normalizer(obs) if self.obs_normalizer else obs
        if self.enable_cache:
            self._cache[evt.time] = result
        return result

    def _get_reward(self, evt: Event, account: Account) -> float:
        equity = account.equity()
        reward = equity / self.last_equity - 1.0
        self.last_equity = equity
        return self.reward_normalizer(reward) if self.reward_normalizer else reward

    def get_signals(self, action, event):
        return {symbol: Signal(rating) for symbol, rating in zip(self.symbols, action)}

    def step(self, action):
        assert self.event is not None
        assert self.account is not None
        signals = self.get_signals(action, self.event)
        logger.debug("time=%s signals=%s", self.event.time, signals)

        orders = self.trader.create_orders(signals, self.event, self.account)
        self.broker.place_orders(orders)
        self.event = self.channel.get()

        if self.event:
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event)
            reward = self._get_reward(self.event, self.account)
            return observation, reward, False, False, {}

        return None, 0.0, True, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.broker.reset()
        self.trader.reset()
        for feature in self.features:
            feature.reset()

        self.channel = self.feed.play_background()

        while True:
            self.event = self.channel.get()
            assert self.event is not None, "feed empty during warmup"
            self.account = self.broker.sync(self.event)
            self.trader.create_orders({}, self.event, self.account)
            observation = self.get_observation(self.event)
            self._get_reward(self.event, self.account)
            if not np.any(np.isnan(observation)):
                return observation, {}

    def render(self):
        pass

    def __str__(self):
        result = (
            f"TradingEnv(\n\tbroker={self.broker}\n\ttrader={self.trader}\n\tfeed={self.feed}"
            f"\n\tfeatures={len(self.features)}"
            f"\n\tobservation_space={self.observation_space}\n\taction_space={self.action_space}"
            "\n)"
        )
        return result


class TraderEnv(gym.Env):
    # pylint: disable=too-many-instance-attributes,unused-argument

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        features: Sequence[Feature],
        feed: Feed,
        rating_symbols: list[str],
        warmup: int = 0,
        broker: Broker | None = None,
    ):
        self.broker: Broker = broker or SimBroker()
        self.channel = EventChannel()
        self.feed = feed
        self.event: Event | None = None
        self.account: Account = self.broker.sync()
        self.symbols = rating_symbols
        self.features = features
        self.last_equity: float = self.account.equity()
        self.obs_normalizer = None
        self.reward_normalizer = None

        action_size = len(rating_symbols)
        obs_size = sum(feature.size() for feature in features)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_size,), dtype=np.float32)

        logger.info("observation_space=%s action_space=%s", self.observation_space, self.action_space)

        self.render_mode = None

    def calc_normalization(self, steps: int):
        obs, _ = self.reset()
        obs_buffer = np.zeros((steps, obs.shape[0]), dtype=np.float32)
        reward_buffer = np.zeros((steps,), dtype=np.float32)
        step = 0
        while step < steps:
            action = self.action_space.sample()
            obs, reward, terminated, _, _ = self.step(action)
            if not terminated:
                obs_buffer[step] = obs
                reward_buffer[step] = reward
            else:
                self.reset()
            step += 1

        obs_norm = obs_buffer.mean(axis=0), obs_buffer.std(axis=0)
        reward_norm = reward_buffer.mean(axis=0).item(), reward_buffer.std(axis=0).item()
        self.obs_normalizer = Normalize(obs_norm)
        self.reward_normalizer = Normalize(reward_norm)

    def get_observation(self, evt: Event, account) -> NDArray[np.float32]:
        data = [feature.calc(evt, account) for feature in self.features]
        obs = np.hstack(data, dtype=np.float32)
        return self.obs_normalizer(obs) if self.obs_normalizer else obs

    def _get_reward(self, evt: Event, account: Account) -> float:
        equity = account.equity()
        reward = equity / self.last_equity - 1.0
        self.last_equity = equity
        return self.reward_normalizer(reward) if self.reward_normalizer else reward

    def account_rebalance(self, account: Account, new_sizes: dict[str, Decimal]) -> list[Order]:
        orders = []
        for symbol, new_size in new_sizes.items():
            old_size = account.get_position_size(symbol)
            order_size = new_size - old_size
            if order_size != Decimal(0):
                order = Order(symbol, order_size)
                orders.append(order)

        return orders

    def get_orders(self, action, event: Event, account: Account) -> list[Order]:
        new_positions = {}
        equity = account.equity()
        for symbol, fraction in zip(self.symbols, action):
            price = event.get_price(symbol)
            if price:
                rel_fraction = fraction / len(self.symbols)
                contract_value = account.contract_value(symbol, Decimal(1), price)
                size = equity * rel_fraction / contract_value
                new_positions[symbol] = Decimal(size).quantize(Decimal(1))
            else:
                new_positions[symbol] = Decimal()

        return self.account_rebalance(account, new_positions)

    def step(self, action):
        assert self.event is not None
        assert self.account is not None
        logger.debug("time=%s action=%s", self.event.time, action)

        orders = self.get_orders(action, self.event, self.account)
        self.broker.place_orders(orders)
        self.event = self.channel.get()

        if self.event:
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event, self.account)
            reward = self._get_reward(self.event, self.account)
            return observation, reward, False, False, {}

        return None, 0.0, True, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.broker.reset()
        for feature in self.features:
            feature.reset()

        self.channel = self.feed.play_background()

        while True:
            self.event = self.channel.get()
            assert self.event is not None, "feed empty during warmup"
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event, self.account)
            self._get_reward(self.event, self.account)
            if not np.any(np.isnan(observation)):
                return observation, {}

    def render(self):
        pass

    def __str__(self):
        result = (
            f"TradingEnv(\n\tbroker={self.broker}\n\tfeed={self.feed}"
            f"\n\tfeatures={len(self.features)}"
            f"\n\tobservation_space={self.observation_space}\n\taction_space={self.action_space}"
            "\n)"
        )
        return result
