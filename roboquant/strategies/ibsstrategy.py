from roboquant.signal import Signal
from roboquant.event import Bar, Event
from roboquant.strategies.strategy import Strategy


class IBSStrategy(Strategy):
    """IBS Strategy implementation.
   - If IBS is below buy threshhold, create a BUY Signal with rating `1.0 - IBS`
   - If IBS is abive sell threshhold, create a SELL Signal with rating `- IBS`
    """

    def __init__(self, buy=0.2, sell=0.8):
        super().__init__()
        self.buy = buy
        self.sell = sell

    def create_signals(self, event: Event) -> list[Signal]:
        result = []
        for asset, item in event.price_items.items():
            if isinstance(item, Bar):
                _, h, l, c, _ = item.ohlcv # noqa: E741
                if h != l:
                    ibs = (c - l) / (h - l)
                    if ibs < self.buy:
                        result.append(Signal(asset, 1.0 - ibs))
                    elif ibs > self.sell:
                        result.append(Signal(asset, -ibs))
        return result

