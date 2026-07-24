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
) -> Account:
    """Start a new run. This can be used for backtesting as well as live-trading.

    Args:
        feed: The feed to use for this run
        strategy: The strategy that you want to validate, use None if you have all the logic in the Trader
        trader: The trader to use, default is the `FlexTrader` if None is provided
        journal: Journal to use to log and/or store progress and metrics, default is None
        broker: The broker you want to use. If None is specified, the `SimBroker` will be used with its default configuration
        timeframe: Optionally limit the run to events within this timeframe. The default is None

    Returns:
        The latest state of the trading account
    """

    broker = broker or SimBroker()
    trader = trader or FlexTrader()

    for event in feed.play(timeframe):
        account = broker.sync(event)
        signals = strategy.create_signals(event) if strategy else []
        orders = trader.create_orders(signals, event, account)
        broker.place_orders(orders)
        if journal:
            journal.track(event, account, signals, orders)

    return broker.sync()

