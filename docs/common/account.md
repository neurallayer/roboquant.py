---
kernelspec:
  name: python3
  display_name: Python 3
---

# Account
(account_def)=
An account reflects the state of the real trading account of the underlying broker. 

Account is the main object owned by the Broker and returned when the `sync()` method is invoked. 
It is also the object returned from the {cl}`run()` function.

```{code-cell} python
import roboquant as rq
account = rq.demo_run()
print(account)
```

It contains available cash, open positions in the portfolio, open orders, available buying power and excuted trades. It doesn't contain closed orders and closed positions. 

:::{note}
When using in a {cl}`Trader` it is important to use `buying_power` and not `cash` to determine the available budget for orders.
:::


There are several helper methods available that can help to further inspect the account.

```{code-cell} python
account.orders_to_dataframe()
account.trades_to_dataframe()
account.pnl()
```