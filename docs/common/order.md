---
kernelspec:
  name: python3
  display_name: Python 3
---

# Order
All orders are limit orders in roboquant. If you want them to behave more like a market order,
you can set a generous `limit` price.

Also the only difference between a BUY and a SELL order is the sign of their `size`. So SELL orders
have a negative `size` and BUY orders a positive `size`.

There are 2 `tif` (time-in-force) values support, "DAY" and "GTC".

```{code-cell} python
from decimal import Decimal
from roboquant import Stock, Order

asset = Stock("AAPL")
buy_order = Order(asset, size = Decimal(10), limit = 200.0, tif="DAY")
sell_order = Order(asset, size = -Decimal(10), limit = 200.0, tif="GTC")
```

Existing orders are orders with a non empty `id`. This `id` is assigned by the broker. These orders
can be found in `account.orders` and contain the open orders only. Only these orders can be cancelled or modified. 

```{code-cell} python
:tags: [remove-input]
order = Order(Stock("ABC"), Decimal(100), limit=50.0, id="1234")
```

```{code-cell} python
assert order.id
modified_order = order.modify(limit=51.0)
cancelled_order = order.cancel()
```

:::{tip}
If you want to track all orders during a run, use the `SignalOrderTracker` journal.
:::
