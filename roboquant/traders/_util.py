import decimal

def round_down(value: float | str | decimal.Decimal | int, ndigits: int) -> decimal.Decimal:
    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_DOWN
        return decimal.Decimal(round(d, ndigits))
