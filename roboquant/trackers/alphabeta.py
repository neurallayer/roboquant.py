from typing import Tuple
import numpy as np

from roboquant.event import Event
from roboquant.timeframe import Timeframe
from .tracker import Tracker


class AlphaBetaTracker(Tracker):
    """Tracks the Alpha and Beta"""

    def __init__(self, price_type="DEFAULT"):
        self.mkt_returns = []
        self.acc_returns = []
        self.last_prices = {}
        self.last_equity = 0.0
        self.init = False
        self.price_type = price_type
        self.start_time = None
        self.end_time = None

    def _get_market_returns(self, prices: dict[str, float]):
        cnt = 0
        result = 0.0
        for symbol in prices.keys():
            if symbol in self.last_prices:
                cnt += 1
                result += prices[symbol] / self.last_prices[symbol] - 1.0
        return result / cnt

    def trace(self, event: Event, account, signals, orders):
        prices = {item.symbol: item.price(self.price_type) for item in event.price_items.values()}
        equity = account.equity
        if self.init:
            self.acc_returns.append(equity / self.last_equity - 1.0)
            self.mkt_returns.append(self._get_market_returns(prices))
        else:
            self.start_time = event.time

        self.end_time = event.time
        self.last_prices.update(prices)
        self.last_equity = equity
        self.init = True

    def alpha_beta(self, risk_free_return=0.0) -> Tuple[float, float]:
        if not self.start_time or not self.end_time:
            return float("nan"), float("nan")

        tf = Timeframe(self.start_time, self.end_time, True)
        mr = np.asarray(self.mkt_returns)
        mr_total = np.cumprod(mr + 1.0)[-1]
        mr_total = tf.annualize(mr_total.item())

        ar = np.asarray(self.acc_returns)
        ar_total = np.cumprod(ar + 1.0)[-1]
        ar_total = tf.annualize(ar_total.item())

        beta = np.cov(mr, ar)[0][1] / np.var(mr)
        alpha = ar_total - risk_free_return - beta * (mr_total - risk_free_return)
        return alpha, beta
