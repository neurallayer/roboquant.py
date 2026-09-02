from datetime import datetime
import decimal

from roboquant.common.monetary import Amount
from roboquant.common.signal import Signal


def round_down(value: float | str | decimal.Decimal | int, ndigits: int) -> decimal.Decimal:
    """round a value to a number of decimals but always down (towards zero)"""

    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_DOWN
        return decimal.Decimal(round(d, ndigits))


def get_order_size(
    signal: Signal, price: float, order_amount: Amount, time: datetime, ndigits: int
) -> decimal.Decimal:
    """Calculate the order size based on the signal rating, asset price, order amount.
    Time is used when a conversion is needed between the asset currency and the order amount currency.
    """
    one_contract_value = signal.asset.amount(decimal.Decimal(1), price).convert_to(order_amount.currency, time)
    size = signal.rating * order_amount.value / one_contract_value
    return round_down(size, ndigits)
