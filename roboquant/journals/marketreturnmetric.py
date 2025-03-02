from roboquant.event import Event
from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.journals.metric import Metric
from roboquant.order import Order
from roboquant.signal import Signal


class MarketReturnMetric(Metric):
    """Tracks the performance of the market, aka the price-items found in the event.
    It will calculate the latest return of the market based on the price-items found
    in the event and track the total return of the market.
    """

    def __init__(self, price_type: str="DEFAULT"):
        self._prev_prices: dict[Asset, float] = {}
        self.price_type = price_type
        self._last_total = 1.0

    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        mkt_return = 0.0
        n = 0
        for asset, price in event.get_prices(self.price_type).items():
            if prev_price := self._prev_prices.get(asset):
                mkt_return += price / prev_price - 1.0
                n += 1

            self._prev_prices[asset] = price

        result = 0.0 if n == 0 else mkt_return / n

        self._last_total *= 1.0 + result

        return {"feed/pnl": result, "feed/total_pnl": self._last_total - 1.0}
