from roboquant.timeframe import Timeframe
from roboquant.trackers.tracker import Tracker


class EquityTracker(Tracker):
    """Tracks the time of an event and the equity at that moment.
    If multiple events happen at the same time, only the equity for first one will be registered.
    """

    def __init__(self):
        self.timeline = []
        self.equities = []
        self.__last = None

    def trace(self, event, account, signals, orders):
        if self.__last is None or event.time > self.__last:
            self.timeline.append(event.time)
            self.equities.append(account.equity)
            self.__last = event.time

    def timeframe(self):
        return Timeframe(self.timeline[0], self.timeline[-1], True)

    def pnl(self, annualized=False):
        """Return the profit & loss percentage, optionally annualized from the recorded durtion"""
        pnl = self.equities[-1]/self.equities[0] - 1
        if annualized:
            return self.timeframe().annualize(pnl)
        else:
            return pnl

    def max_drawdown(self):
        max_equity = self.equities[0]
        result = 0.0
        for equity in self.equities:
            if equity > max_equity:
                max_equity = equity

            dd = (equity - max_equity) / max_equity
            if dd < result:
                result = dd

        return result

    def max_gain(self):
        min_equity = self.equities[0]
        result = 0.0
        for equity in self.equities:
            if equity < min_equity:
                min_equity = equity

            gain = (equity - min_equity) / min_equity
            if gain > result:
                result = gain

        return result
