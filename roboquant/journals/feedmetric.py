from roboquant.journals.metric import Metric


class MarketReturnMetric(Metric):
    """Tracks the performance of the market, aka the price-items found in the event.
    It will calculate the latest return of the market based on the price-items found
    in the event and tracks the total return of the market.
    """

    def __init__(self, price_type="DEFAULT"):
        self._prev_prices = {}
        self.price_type = price_type
        self._last_total = 1.0

    def calc(self, event, account, signals, orders):
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
