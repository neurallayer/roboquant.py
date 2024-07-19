from abc import ABC, abstractmethod
from datetime import timedelta
from decimal import Decimal
import logging
from typing import Callable, Sequence
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.spaces.space import Space
import numpy as np
from numpy.typing import NDArray
from roboquant.account import Account

from roboquant.asset import Asset
from roboquant.brokers.simbroker import SimBroker
from roboquant.event import Event
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.order import Order
from roboquant.ml.features import Feature
from roboquant.timeframe import Timeframe


register(id="roboquant/StrategyEnv-v0", entry_point="roboquant.ml.envs:StrategyEnv")
logger = logging.getLogger(__name__)


class ActionTransformer(ABC):
    """Transforms an action into orders"""

    @abstractmethod
    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]: ...

    @abstractmethod
    def get_action_space(self) -> Space: ...


class OrderMaker(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, assets: list[Asset], order_valid_for=timedelta(days=3)):
        super().__init__()
        self.assets = assets
        self.order_valid_for = order_valid_for

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        orders = []
        equity = account.equity_value()
        assets = {o.asset for o in account.orders}
        bp = account.buying_power.value
        for asset, fraction in zip(self.assets, actions):
            if asset in assets:
                continue
            if price := event.get_price(asset):
                rel_fraction = fraction / len(self.assets)
                contract_value = asset.contract_value(Decimal(1), price)
                if size := round(equity * rel_fraction / contract_value):
                    order = Order(asset, size, price)
                    required_bp = account.required_buying_power(order).value
                    if required_bp < bp:
                        orders.append(order)
                        bp -= required_bp
        return orders

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.assets),), dtype=np.float32)


class OrderWithLimitsMaker(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, assets: Sequence[Asset]):
        super().__init__()
        self.assets = list(assets)

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        orders = []
        equity = account.equity_value()
        bp = account.buying_power.value
        for asset, (fraction, limit_perc) in zip(self.assets, actions.reshape(-1, 2)):
            price = event.get_price(asset)
            if price:
                rel_fraction = fraction / (10 * len(self.assets))
                contract_value = asset.contract_value(Decimal(1), price)
                size = round(equity * rel_fraction / contract_value)
                limit = price * (1.0 + limit_perc / 100.0)
                required = abs(asset.contract_value(size, limit))
                if size and required < bp:
                    if order := next((o for o in account.orders if o.asset == asset), None):
                        orders.append(order.cancel())
                    new_order = Order(asset, size, limit)
                    orders.append(new_order)
                    bp -= required
        return orders

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.assets) * 2,), dtype=np.float32)


class Action2Orders(ActionTransformer):
    """Transforms an action into orders"""

    def __init__(self, assets: list[Asset], price_type="DEFAULT", order_valid_till=timedelta(days=5)):
        super().__init__()
        self.assets = assets
        self.price_type = price_type
        self.order_valid_till = order_valid_till

    def _rebalance(self, account: Account, new_sizes: dict[Asset, Decimal], event: Event) -> list[Order]:
        orders = []
        for asset, new_size in new_sizes.items():
            old_size = account.get_position_size(asset)
            order_size = new_size - old_size
            limit = event.get_price(asset, self.price_type)
            if order_size and limit:
                order = Order(asset, order_size, limit)
                orders.append(order)

        return orders

    def get_orders(self, actions: NDArray, event: Event, account: Account) -> list[Order]:
        new_positions = {}
        equity = account.equity_value()
        for asset, fraction in zip(self.assets, actions):
            if price := event.get_price(asset):
                rel_fraction = fraction / len(self.assets)
                contract_value = asset.contract_value(Decimal(1), price)
                size = equity * rel_fraction / contract_value
                new_positions[asset] = Decimal(size).quantize(Decimal(1))
            else:
                new_positions[asset] = Decimal()

        return self._rebalance(account, new_positions, event)

    def get_action_space(self) -> Space:
        return spaces.Box(-1.0, 1.0, shape=(len(self.assets),), dtype=np.float32)


class TradingEnv(gym.Env):
    # pylint: disable=too-many-instance-attributes,unused-argument

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        feed: Feed,
        obs_feature: Feature,
        reward_feature: Feature,
        action_transformer: ActionTransformer,
        broker: SimBroker | None = None,
        timeframe: Timeframe | None = None,
        journal_factory: Callable[[str], Journal] | None = None
    ):
        self.broker: SimBroker = broker or SimBroker()
        self.channel = EventChannel()
        self.feed = feed
        self.action_transformer = action_transformer
        self.event: Event | None = None
        self.account: Account = self.broker.sync()
        self.obs_feature = obs_feature
        self.reward_feature = reward_feature
        self.timefame = timeframe
        self.journal_factory = journal_factory
        self.journal: Journal | None = None
        self.epoch = 0

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

        if self.journal:
            self.journal.track(self.event, self.account, orders)

        self.event = self.channel.get()

        if self.event:
            self.account = self.broker.sync(self.event)

            if self.account.equity_value() < 0.0:
                logger.info("Account equity < 0, done=True")
                return None, 0.0, True, False, {}

            observation = self.get_observation(self.event, self.account)
            reward = self.get_reward(self.event, self.account)
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
