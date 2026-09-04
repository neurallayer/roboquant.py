---
kernelspec:
  name: python3
  display_name: Python 3
---

# Order

(order_def)=
## Overview
All new orders are either a market- or limit-order in roboquant.

Also, the only difference between a BUY and a SELL order is the sign of their `size`. So SELL orders
have a negative `size` and BUY orders a positive `size`.

There are 2 `tif` (time-in-force) values supported, namely "DAY" and "GTC".

:::{note} history
Originally *roboquant* had support for different order types. However, as it turned out the
way brokers and exchanges implement these more advanced order types differs greatly.

So switching from back testing to live trading introduced different behavior in the way orders
were processed. Also, a lot of complex logic was required to map between the broker order types
and the roboquant order types.

So it was decided to use market- and limit-orders only. Complex order types
can still be implemented as: 
- part of a Trader implementation. For example, automatically create a SELL order
  once the PNL of an open position reaches a certain value (simulate STOP-LOSS functionality) 
- subclass the Order and handled by a custom Broker implementation.
- add info attributes to an Order instance and handled by a custom Broker implementation.
:::

## API
All the attributes of an order are:

```python
class Order:
    """
    A trading order for a particular asset. Orders are immutable.

    Each order has a mandatory `size` and a `limit` price. Orders with a positive `size`
    are buy orders, and with a negative `size` are sell orders.

    The `gtd` (good till date) is optional, and if not set implies the order is valid
    for the DAY. The `info` can hold any arbitrary properties set on the order.

    The `id`, `fill` and `time` properties are managed by the {cl}`Broker`.
    """

    asset: Asset
    """The underlying asset of this order."""

    size: Decimal
    """The size (number of contracts) of the order.
    Positive size for buy orders, negative size for sell orders.
    """

    limit: float | None = None
    """The limit price of the order, denoted in the currency of the asset.
    The limit price is the maximum price you are willing to pay for a buy order,
    or the minimum price you are willing to accept for a sell order.
    Make sure to set the limit price in the currency of the asset
    and not include more decimal places than supported by the broker.
    """

    tif: Literal["GTC", "DAY"] = "DAY"
    """The time in force of the order.
    `GTC` = Good Till Cancelled, `DAY` = valid for a day only.
    """

    info: dict[str, Any] | None = None
    """Any additional information about the order.
    Enables to pass additional information to the broker if required.
    """

    id: str = ""
    """The unique id of the order. This is set by the broker only and should not be updated elsewhere.
    The id is an empty string for new orders and set to a non-empty string when the order is placed with the broker.
    The id is used to identify the order when modifying or cancelling it.
    """

    fill: Decimal = Decimal()
    """The filled size of the order, set by the broker only. Just like the size, positive for buy orders,
    negative for sell orders. So the remaining size is `size - fill`"""

    time: datetime | None = None
    """Time when was the order placed at the exchange.
    So typically the first trading day after the order was submitted
    to the broker.
    """

```

## Initial Orders
Creating a new initial order is simple. One thing to be aware of is that order sizes
are of the type `Decimal`.

```{code-cell} python
from decimal import Decimal
from roboquant import Stock, Order

asset = Stock("AAPL")
buy_mkt_order = Order(asset, size = Decimal(10), tif="DAY")
sell_limit_order = Order(asset, size = -Decimal(10), limit = 200.0, tif="GTC")
```

## Cancel & Modify orders
Existing orders are orders with a non-empty `id`. This `id` is assigned by the broker when placed at that broker.
These are the orders that are found in `account.orders`. Only these type of orders can be canceled or modified. 

If you try set a limit on a market-order an Error will be raised.

```{code-cell} python
:tags: [remove-input]
from roboquant import Account, SimBroker
account = Account.empty()
broker = SimBroker()
order = Order(Stock("ABC"), Decimal(100), limit=50.0, id="1234")
```

```{code-cell} python
assert order.id
modified_order = order.modify(limit=51.0)
cancelled_order = order.cancel()
```

Or to cancel all open orders:

```{code-cell} python
cancellations = [order.cancel() for order in account.orders]
broker.place_orders(cancellations)
```

## Decimals
When creating an order, it is important two realize two things regarding
the number of decimal places:

1. Number of decimals (sometimes referred to as ndigits in Python) for the **order size**
   that is allowed by the broker. For example Forex, Crypto or fractional equity trading all allow for
   fractional sizes.
2. Number of decimals allowed for setting the **limit price**.

There is `round_number` method that perform the rounding (by default rounding down) and return the rounded Decimal: 

```{code-cell} python
from roboquant.traders._util import round_number

# For stocks that don't allow fractional sizes
print(round_number(100.123, "1"))

# For crypto that allow for 0.05 increments
print(round_number(100.123, "0.05"))
```

Most {cl}`Trader` implementations that come with *roboquant* allow you to configure this behavior (named `step_size`) and if you
want to use them in Forex or Crypto, you likely will have to change the defaults.

:::{tip} Market Orders can be dangerous
Roboquant supports market orders, and it might be tempting to
use these order types for everything. But in algo-trading, market
orders introduce more risk. 

For example if there is any data quality issue with a {cl}`Feed` and prices
might be way off, there is no protection when placing the corresponding orders.

Imagine what happens to the order-size when the {cl}`Trader` thinks BTC is worth
500 USD while the real market value is 50_000 USD. And if this is a market-order,
there is nothing stopping this order from being filled (if enough buying power)

So better to use limit orders and get some extra safety. 
:::