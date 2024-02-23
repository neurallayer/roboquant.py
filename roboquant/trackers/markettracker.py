from roboquant.timeframe import Timeframe


class MarketTracker:

    class __Entry:
        """Keeps track of the market returns of a single symbol"""

        __slots__ = "start_time", "end_time", "start_price", "end_price"

        def __init__(self, time, price):
            self.start_time = time
            self.start_price = price
            self.end_time = time
            self.end_price = price

        def weighted(self):
            rate = self.end_price / self.start_price - 1.0
            return rate * self.duration

        @property
        def duration(self):
            return (self.end_time - self.start_time).total_seconds()

    def __init__(self):
        self.market_returns = {}
        self.price_type = "DEFAULT"

    def trace(self, event, account, signals, orders):
        for symbol, item in event.price_items.items():
            price = item.price(self.price_type)
            if mr := self.market_returns.get(symbol):
                mr.end_time = event.time
                mr.end_price = price
            else:
                self.market_returns[symbol] = self.__Entry(event.time, price)

    def timeframe(self):
        start = min([v.start_time for v in self.market_returns.values()])
        end = max([v.end_time for v in self.market_returns.values()])
        return Timeframe(start, end, True)

    def get_market_return(self):
        mr = [v for v in self.market_returns.values()]
        total = sum(v.weighted() for v in mr)
        sum_weights = sum(v.duration for v in mr)
        avg_return = total / sum_weights if sum_weights != 0.0 else float("NaN")
        tf = self.timeframe()
        if tf:
            return tf.annualize(avg_return)
        else:
            return 0.0
