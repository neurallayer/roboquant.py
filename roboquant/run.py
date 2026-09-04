from typing import NoReturn
import logging

from roboquant.common.account import Account
from roboquant.common.asset import Asset
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.common.timeframe import Timeframe
from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.strategies.strategy import Strategy
from roboquant.traders.simpletrader import SimpleTrader
from roboquant.traders.trader import Trader
from roboquant.feeds.yahoofeed import YahooFeed
from roboquant.strategies.ema_crossover import EMACrossover

logger = logging.getLogger(__name__)

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
    trader = trader or _derive_simple_trader(feed.assets(), broker.sync())

    try:
        for event in feed.play(timeframe):
            account: Account = broker.sync(event)
            signals: list[Signal] = strategy.create_signals(event) if strategy else []
            orders : list[Order]= trader.create_orders(signals, event, account)
            broker.place_orders(orders)
            if journal:
                journal.track(event, account, signals, orders)
    except __StopRun as e:
        logger.warning("early stop of the run: %s", e.message)

    return broker.sync()



class __StopRun(Exception):

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

def stop_run(message: str = "") -> NoReturn:
    """Raised an exception that causes the run to be stopped while
    still regularly returning the account object.

    Optionally provide a message that will be part of the logging.
    """
    raise __StopRun(message)


def _derive_simple_trader(assets: list[Asset], account: Account) -> Trader:
    """Derive SimpleTrader settings from provided list of assets
    and broker account
    """

    if not assets:
        return SimpleTrader()

    n = len(assets)
    max_positions = min(100, n)
    return SimpleTrader(max_positions)

def demo_run(journal: Journal | None = None) -> Account:
    """Small demo run for testing purposes.
    Optional a journal can be provided.
    """
    feed = YahooFeed.us_stocks_10(start_date="2022-01-01", end_date="2026-01-01")
    strategy = EMACrossover()
    return run(feed, strategy, journal=journal)
