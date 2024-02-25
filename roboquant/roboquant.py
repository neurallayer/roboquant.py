from roboquant.account import Account
from roboquant.strategies.strategy import Strategy
from roboquant.brokers.broker import Broker
from roboquant.feeds.feed import Feed
from roboquant.traders.trader import Trader
from roboquant.journals.journal import Journal
from roboquant.feeds.eventchannel import EventChannel
from .timeframe import Timeframe
from roboquant.brokers.simbroker import SimBroker
from roboquant.traders.flextrader import FlexTrader
from roboquant.feeds.feedutil import play_background


class Roboquant:
    """The engine of roboquant algo-trading."""

    __slots__ = "strategy", "trader", "broker"

    def __init__(
        self,
        strategy: Strategy,
        trader: Trader | None = None,
        broker: Broker | None = None,
    ) -> None:
        """
        Create a new instance of roboquant

        Args:
        - strategy: your strategy that you want to use, this is the only mandatory argument
        - trader: the trader that you want to use. If None is specified, the `FlexTrader` will be used with its
        default settings
        - broker: the broker that you want to use. If None is specified, the `SimBroker` will be used with its
        default settings
        """
        self.strategy: Strategy = strategy
        self.broker: Broker = broker or SimBroker()
        self.trader: Trader = trader or FlexTrader()

    def run(
        self,
        feed: Feed,
        journal: Journal | None = None,
        timeframe: Timeframe | None = None,
        capacity: int = 10,
        heartbeat_timeout: float | None = None,
    ) -> Account:
        """Start a new run and return the account at the end of the run

        Args:

        - feed: the feed to use for this run. This is the only mandatory argument.
        - journal: journal to use to log and/or store progress and metrics, default is None.
        - timeframe: optionally limit the run to events within this timeframe. The default is None, resulting in all
        events in the feed being delivered.
        - capacity: the buffer capacity of the event channel before it starts blocking new events. Default is 10 events.
        - heartbeat_timeout: optionally, a heartbeat will be generated if no other events are received within the specified
        timeout in seconds. The default is None.
        """

        channel = EventChannel(timeframe, capacity)
        play_background(feed, channel)

        while event := channel.get(heartbeat_timeout):
            signals = self.strategy.create_signals(event)
            account = self.broker.sync(event)
            orders = self.trader.create_orders(signals, event, account)
            self.broker.place_orders(*orders)
            if journal:
                journal.track(event, account, signals, orders)

        return self.broker.sync()
