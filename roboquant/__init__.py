"""
The `roboquant` package contains the `run` method and imports
many useful classes and functions form sub-packages.
"""

import platform
from importlib.metadata import version

from roboquant import brokers, common, feeds, journals, strategies, traders, util
from roboquant.brokers import SimBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset, Crypto, Forex, Option, Stock
from roboquant.common.event import Bar, Event, PriceItem, Quote, TradePrice
from roboquant.common.monetary import EUR, USD, Amount, Currency, Wallet
from roboquant.common.order import Order
from roboquant.common.position import Position
from roboquant.common.signal import Signal, SignalType
from roboquant.common.timeframe import Timeframe, utcnow
from roboquant.common.timeseries import TimeSeries
from roboquant.common.trade import Trade
from roboquant.feeds import Feed
from roboquant.run import demo_run, run, stop_run
from roboquant.strategies import Strategy
from roboquant.util import Report, indicators, set_dark_style, set_light_style

__version__ = version("roboquant")

def info():

    msg = r"""             _______
            | $   $ |         Roboquant v__version__
            |   o   |         Python v__python__
            |_[___]_|         __system__
        ___ ___|_|___ ___
       ()___)       ()___)
      /  / |         | \  \
     (___) |_________| (___)
      | |   __/___\__   | |
      /_\  |_________|  /_\
     // \\  |||   |||  // \\
     \\ //  |||   |||  \\ //
           ()__) ()__)
           ///     \\\
        __///_     _\\\__
       |______|   |______|"""

    msg = msg.replace("__version__", __version__)
    msg = msg.replace("__python__", platform.python_version())
    uname = platform.uname()
    sys_info = f"{uname.system} v{uname.release}"
    msg = msg.replace("__system__", sys_info)
    print(msg, flush=True)

__all__ = [
    "brokers",
    "feeds",
    "journals",
    "strategies",
    "traders",
    "Feed",
    "Strategy",
    "Account",
    "Position",
    "Event",
    "PriceItem",
    "Bar",
    "TradePrice",
    "Quote",
    "Signal",
    "SignalType",
    "Order",
    "Amount",
    "Wallet",
    "Asset",
    "Stock",
    "Crypto",
    "Forex",
    "Option",
    "Trade",
    "run",
    "stop_run",
    "demo_run",
    "indicators",
    "util",
    "common",
    "set_dark_style",
    "set_light_style",
    "Report",
    "SimBroker",
    "Timeframe",
    "TimeSeries",
    "Currency",
    "USD",
    "EUR",
    "utcnow",
]
