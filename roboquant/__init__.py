"""
The `roboquant` package contains the `run` method and a number of shared classes
like `Account`, `Asset` and `Event`.
"""

from roboquant import brokers
from roboquant import feeds
from roboquant import journals
from roboquant import strategies
from roboquant import traders
from roboquant.common import monetary
from roboquant import util

from .common.portfolio import Position, Portfolio
from .common.account import Account
from .common.event import Event, PriceItem, Bar, TradePrice, Quote
from .common.signal import Signal, SignalType
from .common.order import Order
from .common.trade import Trade
from .common.monetary import Amount, Wallet
from .common.asset import Asset, Stock, Crypto, Forex, Option
from .run import run
from .common.timeframe import Timeframe, utcnow
from .common.timeseries import TimeSeries
from roboquant.util import indicators

from importlib.metadata import version
import platform

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
    "Account",
    "Position",
    "Portfolio",
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
    "indicators",
    "util",
    "monetary",
    "Timeframe",
    "TimeSeries",
    "utcnow",
]
