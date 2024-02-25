
from roboquant.trackers.metric import Metric


class EquityMetric(Metric):
    """Calculates the following equity related metrics:
    - `value` equity value itself
    - `mdd` max drawdown
    - `pnl` since the previous step in the run
    - `total_pnl` since the beginning of the run
    """

    def __init__(self):
        self.max_drawdown = 0.0
        self.max_gain = 0.0
        self.prev_equity = None
        self.max_equity = -10e10
        self.min_equity = 10e10

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        equity = account.equity
        return {
            "equity/value": equity,
            "equity/max_drawdown": self.__get_max_drawdown(equity),
            "equity/max_gain": self.__get_max_gain(equity),
            "equity/pnl": self.__get_pnl(equity),
        }

    def __get_pnl(self, equity):
        if self.prev_equity is None:
            self.prev_equity = equity

        result = equity / self.prev_equity - 1.0
        self.prev_equity = equity
        return result

    def __get_max_drawdown(self, equity) -> float:
        if equity > self.max_equity:
            self.max_equity = equity

        drawdown = equity / self.max_equity - 1.0
        if drawdown < self.max_drawdown:
            self.max_drawdown = drawdown

        return self.max_drawdown
    
    def __get_max_gain(self, equity) -> float:
        if equity < self.min_equity:
            self.min_equity = equity

        gain = equity / self.min_equity - 1.0
        if gain > self.max_gain:
            self.max_gain = gain

        return self.max_gain
