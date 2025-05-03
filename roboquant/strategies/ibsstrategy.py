from roboquant.signal import Signal
from roboquant.event import Bar, Event
from roboquant.strategies.strategy import Strategy


class IBSStrategy(Strategy):
    """Internal Bar Strength indicator based-strategy, that follows the following rules:
    - If IBS indicator is below the buy-threshold, create a BUY Signal with a rating of `1.0 - IBS`
    - If IBS indicator is above the sell-threshold, create a SELL Signal with a rating of `- IBS`

    So this is a mean-reversion strategy that uses the IBS indicator to identify oversold or overbought opportunities.
    """

    def __init__(self, buy_threshold: float = 0.2, sell_threshold: float = 0.8):
        super().__init__()
        self.__buy = buy_threshold
        self.__sell = sell_threshold

    def create_signals(self, event: Event) -> list[Signal]:
        result: list[Signal] = []
        for asset, item in event.price_items.items():
            if isinstance(item, Bar):
                _, H, L, C, _ = item.ohlcv
                if H != L:
                    ibs = (C - L) / (H - L)
                    if ibs < self.__buy:
                        result.append(Signal(asset, 1.0 - ibs))
                    elif ibs > self.__sell:
                        result.append(Signal(asset, -ibs))
        return result
