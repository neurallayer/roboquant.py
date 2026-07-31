from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal
import sys
from typing import Tuple, override

import numpy as np

from roboquant.common.account import Account
from roboquant.common.portfolio import Position
from roboquant.common.asset import Asset
from roboquant.common.event import Bar, Event
from roboquant.common.monetary import USD, Amount, Wallet
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.util.buffer import OHLCVBuffer


class Metric(ABC):
    """Metric calculates zero or more values during each step of a run.
    They can be used for example in the MetricsJournal.
    """

    @abstractmethod
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        """Calculate zero or more metrics and return the result as a dictionary. The dictionary should not be modified
        after it is returned. The keys in the dictionary should be unique and not conflict with other metrics.

        Args:
            event: The event to calculate metrics for.
            account: The account to calculate metrics for.
            signals: The signals to calculate metrics for.
            orders: The orders to calculate metrics for.

        Returns:
            The result of the calculations.
        """
        ...


class PNLMetric(Metric):
    """Calculates the following PNL related metrics:
    - `equity` value
    - `mdd` max drawdown
    - `new` pnl since the previous step in the run
    - `unrealized` pnl in the open positions
    - `realized` pnl
    - `total` pnl
    """

    def __init__(self) -> None:
        super().__init__()
        self.max_drawdown: float = 0.0
        self.max_gain: float = 0.0
        self.first_equity: float | None = None
        self.prev_equity: float | None = None
        self.max_equity: float = sys.float_info.min
        self.min_equity: float = sys.float_info.max

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
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

    def __get_pnl_values(self, equity: float, account: Account) -> tuple[float, float, float]:
        if self.first_equity is None:
            self.first_equity = equity

        unrealized = account.convert(account.portfolio.unrealized_pnl())
        total = equity - self.first_equity
        realized = total - unrealized
        return total, realized, unrealized

    def __get_new_pnl(self, equity: float) -> float:
        if self.prev_equity is None:
            self.prev_equity = equity

        result = equity / self.prev_equity - 1.0
        self.prev_equity = equity
        return result

    def __get_max_drawdown(self, equity: float) -> float:
        self.max_equity = max(equity, self.max_equity)
        drawdown = equity / self.max_equity - 1.0
        self.max_drawdown = min(drawdown, self.max_drawdown)
        return self.max_drawdown

    def __get_max_gain(self, equity: float) -> float:
        self.min_equity = min(equity, self.min_equity)
        gain = equity / self.min_equity - 1.0
        self.max_gain = max(gain, self.max_gain)
        return self.max_gain


class IndicatorMetric(Metric):

    def __init__(self, asset: Asset, timeperiod: int):
        self.asset = asset
        self.timeperiod = timeperiod
        self.buffer = OHLCVBuffer(timeperiod)

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        item = event.price_items.get(self.asset)
        if isinstance(item, Bar):
            if self.buffer.append(item.ohlcv):
                return self._calc(self.buffer)
        return {}

    @abstractmethod
    def _calc(self, buffer: OHLCVBuffer) -> dict[str, float]:
        ...


class PriceMetric(Metric):
    """Tracks the price and volume of for one or more assets found in the event."""

    def __init__(self, *assets: Asset, price_type: str = "DEFAULT", volume_type: str = "DEFAULT") -> None:
        """Initialize PriceMetric with specified symbols and price/volume types.
        Args:
            *assets: Variable length argument list of assets to track. If none are provided,
            all encountered assets will be included.
            price_type: Type of price to use for calculations. Defaults to "DEFAULT".
            volume_type: Type of volume to use for calculations. Defaults to "DEFAULT".
        Returns:
            None
        Examples:
            >>> metric = PriceMetric("AAPL", "MSFT", price_type="CLOSE")
        """

        super().__init__()
        self.assets = assets
        self.price_type = price_type
        self.volume_type = volume_type

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        result: dict[str, float] = {}
        for asset, item in event.price_items.items():
            if asset in self.assets or not self.assets:
                symbol = asset.symbol
                prefix = f"item/{symbol.lower()}"
                result[f"{prefix}/price"] = item.price(self.price_type)
                result[f"{prefix}/volume"] = item.volume(self.volume_type)

        return result


@dataclass(slots=True)
class RunMetric(Metric):
    """
    Calculates a number of basic metrics of a run:
    - total number of events in the run
    - total number of items in the events
    - total number of signals generated
    - total number of orders created
    """

    events: int = 0  # Total number of events processed
    items: int = 0   # Total number of items processed
    orders: int = 0  # Total number of orders processed
    signals: int = 0 # Total number of signals processed

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        """
        Update the metrics based on the provided event, account, signals, and orders.

        Args:
            event: The event containing items to be processed.
            account: The account information (not used in this method).
            signals: The list of signals generated.
            orders: The list of orders generated.

        Returns:
            A dictionary with the updated metrics.
        """
        self.items += len(event.items)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)

        return {
            "run/items": self.items,
            "run/signals": self.signals,
            "run/orders": self.orders,
            "run/events": self.events
        }


class MarketMetric(Metric):
    """Calculates the market PNL by acquiring the same amount of all assets and sum their individual PNL performance.
    So this metrics reflects the long only performance of the market.
    """

    def __init__(self, initial_amount: Amount = USD(1_000.0), price_type: str = "DEFAULT") -> None:
        self.initial_amount = initial_amount
        self.positions: dict[Asset, Position] = {}
        self.price_type = price_type

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        for asset, item in event.price_items.items():
            price = item.price(self.price_type)

            if asset not in self.positions:
                converted_value = self.initial_amount.convert_to(asset.currency, event.time)
                size = Decimal(converted_value / price)
                self.positions[asset] = Position(size, price, price)
            else:
                position = self.positions[asset]
                self.positions[asset] = replace(position, mkt_price = price)

        if not self.positions:
            return {
                "market/pnl" : 0.0,
                "market/avg_pnl" : 0.0,
                "market/pnl_pct" : 0.0
            }

        # Calculate the total PNL for all positions
        w = Wallet()
        for asset, position in self.positions.items():
            w += asset.amount(position.size, position.mkt_price - position.avg_price)

        pnl = account.convert(w)
        avg_pnl = pnl / len(self.positions)
        pnl_pct = pnl / ( account.convert(self.initial_amount) * len(self.positions))

        return {
            "market/pnl" : pnl,
            "market/avg_pnl" : avg_pnl,
            "market/pnl_pct" : pnl_pct
        }


class AssetMetric(Metric):
    """Tracks the combined performance of the assets found in the feed. It will:
    - calculate the latest pnl of the assets compared to their previous price
    - track the combined total pnl till this point in time
    - count the number of unique assets so for encountered
    """

    def __init__(self, price_type: str = "DEFAULT") -> None:
        self._prev_prices: dict[Asset, float] = {}
        self.price_type: str = price_type
        self._last_total: float = 1.0

    @override
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        mkt_return: float = 0.0
        n: int = 0
        for asset, item in event.price_items.items():
            price = item.price(self.price_type)
            if prev_price := self._prev_prices.get(asset):
                mkt_return += price / prev_price - 1.0
                n += 1

            self._prev_prices[asset] = price

        result = 0.0 if n == 0 else mkt_return / n

        self._last_total *= 1.0 + result

        return {
            "feed/pnl": result,
            "feed/total_pnl": self._last_total - 1.0,
            "feed/assets": len(self._prev_prices)
        }


class AlphaBeta(Metric):
    """
    Calculate the alpha and beta metric over a given window size.
    """

    def __init__(self, window_size: int, price_type: str = "DEFAULT", risk_free_return: float = 0.0) -> None:
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

    def __update(self, equity: float, prices: dict[Asset, float]) -> None:
        self.__last_equity = equity
        self.__last_prices.update(prices)

    @override
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
