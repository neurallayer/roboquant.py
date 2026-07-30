from dataclasses import dataclass
from datetime import datetime
from typing import override

from roboquant.common.asset import Asset
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.journals.journal import Journal
from roboquant.common.timeframe import Timeframe
from roboquant.common.timeseries import TimeSeries

@dataclass
class SignalOrderTracker(Journal):
    """Tracks the generated signals and created orders from each step
    """

    def __init__(self) -> None:
        self.orders: dict[datetime, list[Order]] = {}
        self.signals: dict[datetime, list[Signal]] = {}

    @override
    def track(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> None:
        self.signals[event.time] = signals
        self.orders[event.time] = orders

    def get_order_limits(self, asset: Asset, timeframe: Timeframe | None = None) -> TimeSeries:
        """Get the order limits for an asset within the given timeframe. Cancel or modify
            orders are ignored.
        """
        timeline : list[datetime]= []
        data : list[float] = []
        for time, orders in self.orders.items():
            if not timeframe or time in timeframe:
                tmp = [order.limit for order in orders if order.asset == asset and not order.id]
                if tmp:
                    timeline.append(time)
                    data.append(tmp[0])
        return TimeSeries(asset.symbol + "-order-limits", timeline, data)

    def get_order_sizes(self, asset: Asset, timeframe: Timeframe | None = None) -> TimeSeries:
        """Get the order size for an asset within the given timeframe. Cancel or modify
        orders are ignored.
        """
        timeline : list[datetime]= []
        data : list[float] = []
        for time, orders in self.orders.items():
            if not timeframe or time in timeframe:
                tmp = [float(order.size) for order in orders if order.asset == asset and not order.id]
                if tmp:
                    timeline.append(time)
                    data.append(tmp[0])
        return TimeSeries(asset.symbol + "-order-sizes", timeline, data)

    def get_signal_ratings(self, asset: Asset, timeframe: Timeframe | None = None) -> TimeSeries:
        """Get the signal ratings for an asset within the given timeframe.
        If there is more than one signal at a given time, it returns
        the rating of the first signal.
        """
        timeline : list[datetime]= []
        data : list[float] = []
        for time, signals in self.signals.items():
            if not timeframe or time in timeframe:
                tmp = [signal.rating for signal in signals if signal.asset == asset]
                if tmp:
                    timeline.append(time)
                    data.append(tmp[0])
        return TimeSeries(asset.symbol + "-signal-ratings", timeline, data)

