from roboquant.strategies.strategy import Strategy


class NOPStrategy(Strategy):
    """This strategy will never generate a signal.

    This is useful if you decide to implement all the trading logic in a Trader and don't require a Strategy.
    """

    def create_signals(self, event):
        """Always return an empty dict"""
        return {}
