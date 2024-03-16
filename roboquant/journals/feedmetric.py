from roboquant.journals.metric import Metric


class FeedMetric(Metric):
    """Tracks the combined performance of the price-items in the event"""

    def __init__(self, price_type="DEFAULT"):
        self._prev_prices = {}
        self.price_type = price_type
        self._last_total = 1.0

    def calc(self, event, account, signals, orders):
        mkt_return = 0.0
        n = 0
        for symbol, price in event.get_prices(self.price_type).items():
            if prev_price := self._prev_prices.get(symbol):
                mkt_return += price / prev_price - 1.0
                n += 1

            self._prev_prices[symbol] = price

        result = 0.0 if n == 0 else mkt_return / n

        new_total = self._last_total * (1.0 + result)
        self._last_total = new_total

        return {"feed/pnl": result, "feed/total_pnl": new_total - 1.0}
