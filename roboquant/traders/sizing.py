from roboquant.event import Bar, Event
from roboquant.feeds.yahoo import YahooFeed


class ETR:
    "Average True Range with a twist, it uses an exponential moving average of the true range"

    def __init__(self, period=10, smoothing=2.0):
        self.history = {}
        self.momentum = 1.0 - (smoothing / (period + 1))

    def add(self, event: Event):
        m = self.momentum

        for symbol, item in event.price_items.items():
            if not isinstance(item, Bar):
                continue
            ohlcv = item.ohlcv
            h, l, c = ohlcv[1], ohlcv[2], ohlcv[3]
            if symbol not in self.history:
                self.history[symbol] = (h - l, c)
            else:
                prev_etr, prev_close = self.history[symbol]
                new_etr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                etr = prev_etr * m + new_etr * (1.0 - m)
                self.history[symbol] = (etr, c)

    def get_value(self, symbol) -> float | None:
        entry = self.history.get(symbol)
        if entry:
            return entry[0]
        return None


if __name__ == "__main__":
    feed = YahooFeed("IBM", "TSLA")
    channel = feed.play_background()
    e = ETR(80, 0.1)
    while evt := channel.get():
        e.add(evt)
        print(e.get_value("IBM"), e.get_value("TSLA"))
