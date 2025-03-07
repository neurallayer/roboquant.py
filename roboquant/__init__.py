"""
The `roboquant` package contains the `run` method and a number of shared classes
like `Account`, `Asset` and `Event`.
"""

__version__ = "1.2.0"

import logging

from roboquant import brokers
from roboquant import feeds
from roboquant import journals
from roboquant import strategies
from roboquant import traders
from roboquant import ml
from .account import Account, Position
from .event import Event, PriceItem, Bar, Trade, Quote
from .signal import Signal, SignalType
from .order import Order
from .monetary import Amount, Wallet
from .asset import Asset, Stock, Crypto, Option
from .run import run
from .timeframe import Timeframe, utcnow

logger = logging.getLogger(__name__)
logger.info("roboquant version=%s", __version__)

__all__ = [
    "brokers",
    "feeds",
    "journals",
    "strategies",
    "traders",
    "ml",
    "Account",
    "Position",
    "Event",
    "PriceItem",
    "Bar",
    "Trade",
    "Quote",
    "Signal",
    "SignalType",
    "Order",
    "Amount",
    "Wallet",
    "Asset",
    "Stock",
    "Crypto",
    "Option",
    "run",
    "Timeframe",
    "utcnow",
]
