from roboquant.account import Account
from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.feeds.eventchannel import EventChannel
from roboquant.feeds.feed import Feed
from roboquant.feeds.feedutil import play_background
from roboquant.journals.journal import Journal
from roboquant.strategies.strategy import Strategy
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.trader import Trader
from .timeframe import Timeframe


def run(
        feed: Feed,
        strategy: Strategy,
        trader: Trader | None = None,
        broker: Broker | None = None,
        journal: Journal | None = None,
        timeframe: Timeframe | None = None,
        capacity: int = 10,
        heartbeat_timeout: float | None = None,
) -> Account:
    """Start a new run. Only the first two parameters, the feed and strategy, are mandatory.
    The other parameters are optional.

    Args:
    - feed: the feed to use for this run
    - strategy: your strategy that you want to use
    - trader: the trader you want to use. If None is specified, the `FlexTrader` will be used with its default settings
    - broker: the broker you want to use. If None is specified, the `SimBroker` will be used with its default settings
    - journal: journal to use to log and/or store progress and metrics, default is None.
    - timeframe: optionally limit the run to events within this timeframe. The default is None
    - capacity: the buffer capacity of the event channel. Default is 10 events.
    - heartbeat_timeout: optionally, a heartbeat will be generated if no other events are received within the specified
    timeout in seconds. The default is None.

    Returns:
    The latest version of the account
    """

    trader = trader or FlexTrader()
    broker = broker or SimBroker()

    channel = EventChannel(timeframe, capacity)
    play_background(feed, channel)

    while event := channel.get(heartbeat_timeout):
        signals = strategy.create_signals(event)
        account = broker.sync(event)
        orders = trader.create_orders(signals, event, account)
        broker.place_orders(orders)
        if journal:
            journal.track(event, account, signals, orders)

    return broker.sync()
