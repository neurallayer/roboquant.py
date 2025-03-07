from roboquant.signal import Signal
from roboquant.event import Bar, Event
from roboquant.strategies.strategy import Strategy


class IBSStrategy(Strategy):
    """Internal Bar Strength indicator based strategy, that follows the following rules:
   - If IBS indicator is below buy-threshhold, create a BUY Signal with rating `1.0 - IBS`
   - If IBS indicator is above sell-threshhold, create a SELL Signal with rating `- IBS`

   So it is a mean-reversion strategy that uses IBS to identify oversold or overbought opprtunities.
    """

    def __init__(self, buy_threshold=0.2, sell_threshold=0.8):
        super().__init__()
        self.__buy = buy_threshold
        self.__sell = sell_threshold

    def create_signals(self, event: Event) -> list[Signal]:
        result = []
        for asset, item in event.price_items.items():
            if isinstance(item, Bar):
                _, h, l, c, _ = item.ohlcv # noqa: E741
                if h != l:
                    ibs = (c - l) / (h - l)
                    if ibs < self.__buy:
                        result.append(Signal(asset, 1.0 - ibs))
                    elif ibs > self.__sell:
                        result.append(Signal(asset, -ibs))
        return result

