import decimal

def round_down(value: float | str | decimal.Decimal | int, ndigits: int) -> decimal.Decimal:
    """round a value to a number of decimals but always down (towards zero)"""

    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_DOWN
        return decimal.Decimal(round(d, ndigits))
