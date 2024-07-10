from typing import Literal

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.strategies.strategy import Strategy


class MultiStrategy(Strategy):
    """Combine one or more signal strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same symbols via the signal_filter:

    - first: in case of multiple signals for the same symbol, the first one wins
    - last:  in case of multiple signals for the same symbol, the last one wins.
    - avg: return the avgerage of the signals. All signals will be ENTRY and EXIT.
    - none: return all signals. This is also the default.
    """

    def __init__(
        self,
        *strategies: Strategy,
        order_filter: Literal["last", "first", "none"] = "none"
    ):
        super().__init__()
        self.strategies = list(strategies)
        self.filter = order_filter

    def create_orders(self, event: Event, account: Account) -> list[Order]:
        orders: list[Order] = []
        for strategy in self.strategies:
            orders += strategy.create_orders(event, account)

        match self.filter:
            case "none":
                return orders
            case "last":
                s = {s.asset: s for s in orders}
                return list(s.values())
            case "first":
                s = {s.asset: s for s in reversed(orders)}
                return list(s.values())

        raise ValueError("unsupported signal filter")
