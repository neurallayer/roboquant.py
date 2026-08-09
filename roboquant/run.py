from typing import NoReturn
import logging

from roboquant.common.account import Account
from roboquant.common.asset import Asset, Crypto, Forex
from roboquant.common.timeframe import Timeframe
from roboquant.brokers.broker import Broker
from roboquant.brokers.simbroker import SimBroker
from roboquant.feeds.feed import Feed
from roboquant.journals.journal import Journal
from roboquant.strategies.strategy import Strategy
from roboquant.traders.flextrader import FlexTrader
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
            account = broker.sync(event)
            signals = strategy.create_signals(event) if strategy else []
            orders = trader.create_orders(signals, event, account)
            broker.place_orders(orders)
            if journal:
                journal.track(event, account, signals, orders)
    except __StopRun as e:
        logger.info("early stop of the run", e.message)
        pass

    return broker.sync()



class __StopRun(Exception):

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

def stop_run(message: str = "") -> NoReturn:
    """Raised an exception that causes the run to be stopped while
    still regulary returning the account object.

    Optionally provide a message that will be part of the logging.
    """
    raise __StopRun(message)


def _derive_simple_trader(assets: list[Asset], account: Account) -> SimpleTrader:
    """Derive SimpleTrader settings from provided list of assets
    and broker account
    """

    if not assets:
        return SimpleTrader()

    n = len(assets)
    max_positions = min(100, n)
    return SimpleTrader(max_positions)



def _derive_flex_trader(assets: list[Asset], account: Account) -> "FlexTrader":
    """Derive FlexTrader settings from provided list of assets
    and broker account
    """

    if not assets:
        return FlexTrader()

    n = len(assets)
    min_order_pct = max(0.01, 1.0/(n * 4))
    max_order_pct = max(0.02, 1.0/(n * 2))
    max_position_pct = max(0.04, 1.0 / n)

    if isinstance(assets[0], Crypto) or isinstance(assets[0], Forex):
        shorting = True
        size_fractions = 6
        limit_rounding = 8
    else:
        shorting = False
        size_fractions = 0
        limit_rounding = 2

    return FlexTrader(
        min_order_pct=min_order_pct,
        max_order_pct=max_order_pct,
        max_position_pct=max_position_pct,
        shorting=shorting,
        size_fractions=size_fractions,
        limit_rounding=limit_rounding
    )


def demo_run() -> Account:
    """Small demo run for testing purposes"""
    feed = YahooFeed.us_stocks_10()
    strategy = EMACrossover()
    return run(feed, strategy)
