from roboquant.account import Account
from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.strategies.strategy import Strategy
from roboquant.timeframe import Timeframe


def run(
    feed: Feed,
    strategy: Strategy,
    journal: Journal | None = None,
    broker: Broker | None = None,
    timeframe: Timeframe | None = None,
    capacity: int = 10,
    heartbeat_timeout: float | None = None
) -> Account:
    """Start a new run.

    Args:
        feed: The feed to use for this run
        strategy: Your strategy that you want to use
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

    while event := channel.get(heartbeat_timeout):
        account = broker.sync(event)
        orders = strategy.create_orders(event, account)
        broker.place_orders(orders)
        if journal:
            journal.track(event, account, orders)

    return broker.sync()
