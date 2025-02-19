from roboquant.account import Account
from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.strategies.strategy import Strategy
from roboquant.timeframe import Timeframe
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.trader import Trader


def run(
    feed: Feed,
    strategy: Strategy | None,
    trader: Trader | None = None,
    journal: Journal | None = None,
    broker: Broker | None = None,
    timeframe: Timeframe | None = None,
    capacity: int = 10,
    heartbeat_timeout: float | None = None
) -> Account:
    """Start a new run. A run can be seen as a simulation of a trading strategy. It will use the provided feed to
    generate events and the strategy to create signals. The trader will convert these signals into orders that will be sent
    to the broker. The broker will execute these orders and update the account accordingly. The journal can be used to log
    and/or store progress and metrics.

    At the end of the run, the latest version of the account will be returned.

    Args:
        feed: The feed to use for this run
        strategy: Your strategy that you want to use
        trader: The trader to use, default is FlexTrader
        journal: Journal to use to log and/or store progress and metrics, default is None
        broker: The broker you want to use. If None is specified, the `SimBroker` will be used with its default settings
        timeframe: Optionally limit the run to events within this timeframe. The default is None
        capacity: The max capacity of the used event channel. Default is 10 events.
        heartbeat_timeout: Optionally, a heartbeat (is an empty event) will be generated if no other events are received
        within the specified timeout in seconds. The default is None. This should normally only be used with live feeds since
        the timestamp used for the heartbeat is the current time.

    Returns:
        The latest version of the account
    """

    broker = broker or SimBroker()
    channel = feed.play_background(timeframe, capacity)
    trader = trader or FlexTrader()

    while event := channel.get(heartbeat_timeout):
        account = broker.sync(event)
        signals = strategy.create_signals(event) if strategy else []
        orders = trader.create_orders(signals, event, account)
        broker.place_orders(orders)
        if journal:
            journal.track(event, account, signals, orders)

    return broker.sync()
