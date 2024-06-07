from abc import ABC, abstractmethod
from datetime import timedelta
from decimal import Decimal
import logging
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.spaces.space import Space
import numpy as np
from numpy.typing import NDArray
from roboquant.account import Account

from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.order import Order
from roboquant.ml.features import Feature
from roboquant.timeframe import Timeframe


register(id="roboquant/StrategyEnv-v0", entry_point="roboquant.ml.envs:StrategyEnv")
logger = logging.getLogger(__name__)


class ActionTransformer(ABC):
    """Transforms an action into orders"""

    @abstractmethod
    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        ...

    @abstractmethod
    def get_action_space(self) -> Space:
        ...


class OrderMaker(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, symbols: list[str], order_valid_for=timedelta(days=3)):
        super().__init__()
        self.symbols = symbols
        self.order_valid_for = order_valid_for

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        orders = []
        gtd = event.time + self.order_valid_for
        equity = account.equity()
        for symbol, fraction in zip(self.symbols, actions):
            price = event.get_price(symbol)
            if price:
                rel_fraction = fraction / len(self.symbols)
                contract_value = account.contract_value(symbol, price)
                size = equity * rel_fraction / contract_value
                order = Order(symbol, size, price, gtd)
                orders.append(order)
        return orders

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.symbols),), dtype=np.float32)


class OrderWithLimitsMaker(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, symbols: list[str]):
        super().__init__()
        self.symbols = symbols

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        orders = []
        gtd = event.time + timedelta(days=3)
        equity = account.equity()
        for symbol, (fraction, limit_perc) in zip(self.symbols, actions.reshape(-1, 2)):
            price = event.get_price(symbol)
            if price:
                rel_fraction = fraction / len(self.symbols)
                contract_value = account.contract_value(symbol, price)
                size = equity * rel_fraction / contract_value
                limit = price * (1.0 + limit_perc/100.0)
                order = Order(symbol, size, limit, gtd)
                orders.append(order)
        return orders

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.symbols)*2,), dtype=np.float32)


class Action2Orders(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, symbols: list[str], price_type="DEFAULT", order_valid_till=timedelta(days=5)):
        super().__init__()
        self.symbols = symbols
        self.price_type = price_type
        self.order_valid_till = order_valid_till

    def _rebalance(self, account: Account, new_sizes: dict[str, Decimal], event: Event) -> list[Order]:
        orders = []
        gtd = event.time + self.order_valid_till
        for symbol, new_size in new_sizes.items():
            old_size = account.get_position_size(symbol)
            order_size = new_size - old_size
            limit = event.get_price(symbol, self.price_type)
            if order_size != Decimal(0) and limit:
                order = Order(symbol, order_size, limit, gtd)
                orders.append(order)

        return orders

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        new_positions = {}
        equity = account.equity()
        for symbol, fraction in zip(self.symbols, actions):
            price = event.get_price(symbol)
            if price:
                rel_fraction = fraction / len(self.symbols)
                contract_value = account.contract_value(symbol, price)
                size = equity * rel_fraction / contract_value
                new_positions[symbol] = Decimal(size).quantize(Decimal(1))
            else:
                new_positions[symbol] = Decimal()

        return self._rebalance(account, new_positions, event)

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.symbols),), dtype=np.float32)


class StrategyEnv(gym.Env):
    # pylint: disable=too-many-instance-attributes,unused-argument

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        feed: Feed,
        obs_feature: Feature,
        reward_feature:  Feature,
        action_transformer: ActionTransformer,
        broker: Broker | None = None,
        timeframe: Timeframe | None = None
    ):
        self.broker: Broker = broker or SimBroker()
        self.channel = EventChannel()
        self.feed = feed
        self.action_transformer = action_transformer
        self.event: Event | None = None
        self.account: Account = self.broker.sync()
        self.obs_feature = obs_feature
        self.reward_feature = reward_feature
        self.timefame = timeframe

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_feature.size(),), dtype=np.float32)
        self.action_space = action_transformer.get_action_space()

        logger.info("observation_space=%s action_space=%s", self.observation_space, self.action_space)

        self.render_mode = None

    def get_observation(self, evt: Event, account) -> NDArray[np.float32]:
        return self.obs_feature.calc(evt, account)

    def get_reward(self, evt: Event, account: Account):
        return self.reward_feature.calc(evt, account)

    def step(self, action):
        assert self.event is not None
        assert self.account is not None
        logger.debug("time=%s action=%s", self.event.time, action)

        orders = self.action_transformer.get_orders(action, self.event, self.account)
        self.broker.place_orders(orders)
        self.event = self.channel.get()

        if self.event:
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event, self.account)
            reward = self.get_reward(self.event, self.account)
            return observation, reward, False, False, {}

        return None, 0.0, True, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.broker.reset()
        self.obs_feature.reset()
        self.reward_feature.reset()

        self.channel = self.feed.play_background(self.timefame)

        while True:
            self.event = self.channel.get()
            assert self.event is not None, "feed empty during warmup"
            self.account = self.broker.sync(self.event)
            observation = self.get_observation(self.event, self.account)
            self.get_reward(self.event, self.account)
            if not np.any(np.isnan(observation)):
                return observation, {}
            logger.info(observation)

    def render(self):
        pass

    def __repr__(self):
        result = (
            f"TradingEnv(\n\tbroker={self.broker}\n\tfeed={self.feed}"
            f"\n\tfeature_size={self.obs_feature.size()}"
            f"\n\tobservation_space={self.observation_space}\n\taction_space={self.action_space}"
            "\n)"
        )
        return result
