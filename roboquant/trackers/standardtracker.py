from datetime import datetime
from ..timeframe import Timeframe
from ..event import Event
from prettytable import PrettyTable
import math
from .tracker import Tracker


class _MarketReturn:
    """Keeps track of the market returns of a single symbol"""

    __slots__ = "start_time", "end_time", "start_price", "end_price"

    def __init__(self, time, price):
        self.start_time = time
        self.start_price = price
        self.end_time = time
        self.end_price = price

    def weighted(self):
        rate = self.end_price / self.start_price - 1.0
        return rate * self.duration

    @property
    def duration(self):
        return (self.end_time - self.start_time).total_seconds()


class _PropertyCalculator:
    """Keeps track of the market returns of a single symbol"""

    __slots__ = "start_time", "end_time", "total"

    def __init__(self):
        self.start_time = None
        self.total = 0
        self.end_time = None

    def add(self, value: int, time: datetime):
        if value != 0:
            self.total += value
            self.end_time = time
            if self.start_time is None:
                self.start_time = time


class _EquityCalculator:
    """Tracks several equity metrics"""

    def __init__(self):
        self.max_equity = -10e9
        self.min_equity = 10e9
        self.mdd = 0.0
        self.max_gain = 0.0
        self.start_equity = float("nan")
        self.end_equity = float("nan")

    def add(self, equity):
        if math.isnan(self.start_equity):
            self.start_equity = equity

        self.end_equity = equity

        if equity > self.max_equity:
            self.max_equity = equity

        if equity < self.min_equity:
            self.min_equity = equity

        dd = (equity - self.max_equity) / self.max_equity
        if dd < self.mdd:
            self.mdd = dd

        gain = (equity - self.min_equity) / self.min_equity
        if gain > self.max_gain:
            self.max_gain = gain


class StandardTracker(Tracker):
    """Tracks a number of key metrics:
    - total, min and max of events, items, ratings, orders and equity
    - drawdown and gain
    - annual performance
    - market performance
    """

    def __init__(self, price_type="DEFAULT"):
        self.properties = {
            "event": _PropertyCalculator(),
            "item": _PropertyCalculator(),
            "rating": _PropertyCalculator(),
            "order": _PropertyCalculator(),
        }
        self.market_returns: dict[str, _MarketReturn] = dict()
        self.price_type = price_type
        self.mddCalculator = _EquityCalculator()
        self.max_positions = 0

    def _update_market_returns(self, event: Event):
        for symbol, item in event.price_items.items():
            price = item.get_price(self.price_type)
            if mr := self.market_returns.get(symbol):
                mr.end_time = event.time
                mr.end_price = price
            else:
                self.market_returns[symbol] = _MarketReturn(event.time, price)

    def get_market_return(self):
        mr = [v for v in self.market_returns.values()]
        total = sum(v.weighted() for v in mr)
        sum_weights = sum(v.duration for v in mr)
        avg_return = total / sum_weights if sum_weights != 0.0 else float("NaN")
        tf = self.timeframe()
        if tf:
            return tf.annualize(avg_return)
        else:
            return 0.0

    def log(self, event, account, ratings, orders):
        t = event.time
        prop = self.properties

        prop["event"].add(1, t)
        prop["item"].add(len(event.items), t)
        prop["rating"].add(len(ratings), t)
        prop["order"].add(len(orders), t)

        if (npositions := len(account.positions)) > self.max_positions:
            self.max_positions = npositions

        self.mddCalculator.add(account.equity)
        self._update_market_returns(event)

    def __repr__(self) -> str:
        if self.properties["event"].total == 0:
            return "no events observed"

        def to_timefmt(time: datetime | None):
            return "-" if time is None else time.strftime("%Y-%m-%d %H:%M:%S")

        pnl = self.annualized_pnl() * 100
        mkt_pnl = self.get_market_return() * 100
        p = PrettyTable(["metric", "value"], align="r", float_format=".2")
        for k, v in self.properties.items():
            p.add_row([f"total {k}s", v.total])
            p.add_row([f"first {k}", to_timefmt(v.start_time)])
            p.add_row([f"last {k}", to_timefmt(v.end_time)], divider=True)

        p.add_row(["max positions", self.max_positions], divider=True)

        p.add_row(["start equity", self.mddCalculator.start_equity])
        p.add_row(["end equity", self.mddCalculator.end_equity])
        p.add_row(["min equity", self.mddCalculator.min_equity])
        p.add_row(["max equity", self.mddCalculator.max_equity], divider=True)

        p.add_row(["max drawdown %", self.mddCalculator.mdd * 100])
        p.add_row(["max gain %", self.mddCalculator.max_gain * 100], divider=True)

        p.add_row(["annual pnl %", pnl])
        p.add_row(["annual mkt %", mkt_pnl])
        return p.get_string()

    def timeframe(self):
        events = self.properties["event"]
        if events.total > 0:
            return Timeframe(events.start_time, events.end_time, inclusive=True)  # type: ignore

    def annualized_pnl(self):
        pnl = self.mddCalculator.end_equity / self.mddCalculator.start_equity - 1.0
        tf = self.timeframe()
        if tf:
            return tf.annualize(pnl)
        else:
            return 0.0
