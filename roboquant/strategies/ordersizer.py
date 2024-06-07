from decimal import Decimal
from roboquant.account import Account


class OrderSizer:

    def __init__(self, account: Account):
        self.account = account
        self.buying_power = account.buying_power
        self.equity = account.equity()
        self.max_order_size = self.equity * 0.2
        self.size_digits = 0

    def is_reduction(self, symbol: str, size: Decimal) -> bool:
        """Determine the kind of change a certain action would have on the position"""
        pos_size = self.account.get_position_size(symbol)
        return abs(size + pos_size) < abs(pos_size)

    def size(self, symbol: str, price: float, percentage: float) -> Decimal:
        contract_price = self.account.contract_value(symbol, price)
        size = Decimal(percentage * self.equity * 0.2 / contract_price)
        size = round(size, self.size_digits)
        if not self.is_reduction(symbol, size):
            contract_value = contract_price * float(size)
            if self.buying_power > contract_value:
                self.buying_power -= contract_value
                return size
            return Decimal()
        return size
