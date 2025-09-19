from typing import Tuple

import numpy as np

from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.event import Event
from roboquant.journals.metric import Metric
from roboquant.order import Order
from roboquant.signal import Signal


class AlphaBeta(Metric):
    """
    Calculate the alpha and beta metric over a given window size.
    """

    def __init__(self, window_size: int, price_type: str = "DEFAULT", risk_free_return: float = 0.0):
        """
        window_size: the rolling window_size to use. The alpha and beta are only calculated once the window is filled.
        price_type: the type of price to use to calculate the market returns, default is "DEFAULT"
        risk_free_return: the risk-free return rate, default is 0.0
        """

        # data stores both portfolio return and market return
        super().__init__()
        self._data = np.ones((2, window_size))
        self.__cnt = 0
        self.__last_prices: dict[Asset, float] = {}
        self.__last_equity: float | None = None
        self.risk_free_return = risk_free_return
        self.price_type = price_type

    def __get_market_value(self, prices: dict[Asset, float]) -> float:
        cnt = 0
        result = 0.0
        for asset in prices.keys():
            if asset in self.__last_prices:
                cnt += 1
                result += prices[asset] / self.__last_prices[asset]
        return 1.0 if cnt == 0 else result / cnt

    def __update(self, equity: float, prices: dict[Asset, float]):
        self.__last_equity = equity
        self.__last_prices.update(prices)

    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        prices = event.get_prices(self.price_type)
        equity = account.equity_value()
        if self.__last_equity is None:
            self.__update(equity, prices)
            return {}

        idx = self.__cnt % len(self._data)
        self._data[0, idx] = equity / self.__last_equity
        self._data[1, idx] = self.__get_market_value(prices)
        self.__update(equity, prices)
        self.__cnt += 1

        if self.__cnt <= self._data.shape[-1]:
            return {}

        alpha, beta = self.alpha_beta()
        return {"perf/alpha": alpha, "perf/beta": beta}

    def alpha_beta(self) -> Tuple[float, float]:
        ar_total, mr_total = np.cumprod(self._data, axis=1)[:, -1]

        beta = np.cov(self._data)[0][1] / np.var(self._data[1])
        alpha = ar_total - self.risk_free_return - beta * (mr_total - self.risk_free_return)
        return alpha, beta
