from roboquant.account import Account
from roboquant.journals.metric import Metric


class PNLMetric(Metric):
    """Calculates the following PNL related metrics:
    - `equity` value
    - `mdd` max drawdown
    - `new` pnl since the previous step in the run
    - `unrealized` pnl in the open positions
    - `realized` pnl
    - `total` pnl
    """

    def __init__(self):
        super().__init__()
        self.max_drawdown = 0.0
        self.max_gain = 0.0
        self.first_equity = None
        self.prev_equity = None
        self.max_equity = -10e10
        self.min_equity = 10e10

    def calc(self, event, account, orders) -> dict[str, float]:
        equity = account.equity_value()

        total, realized, unrealized = self.__get_pnl_values(equity, account)

        return {
            "pnl/equity": equity,
            "pnl/max_drawdown": self.__get_max_drawdown(equity),
            "pnl/max_gain": self.__get_max_gain(equity),
            "pnl/new": self.__get_new_pnl(equity),
            "pnl/total": total,
            "pnl/realized": realized,
            "pnl/unrealized": unrealized,
        }

    def __get_pnl_values(self, equity, account: Account):
        if self.first_equity is None:
            self.first_equity = equity

        unrealized = account.unrealized_pnl_value()
        total = equity - self.first_equity
        realized = total - unrealized
        return total, realized, unrealized

    def __get_new_pnl(self, equity):
        if self.prev_equity is None:
            self.prev_equity = equity

        result = equity / self.prev_equity - 1.0
        self.prev_equity = equity
        return result

    def __get_max_drawdown(self, equity) -> float:
        self.max_equity = max(equity, self.max_equity)
        drawdown = equity / self.max_equity - 1.0
        self.max_drawdown = min(drawdown, self.max_drawdown)
        return self.max_drawdown

    def __get_max_gain(self, equity) -> float:
        self.min_equity = min(equity, self.min_equity)
        gain = equity / self.min_equity - 1.0
        self.max_gain = max(gain, self.max_gain)
        return self.max_gain
