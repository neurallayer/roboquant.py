---
kernelspec:
  name: python3
  display_name: Python 3
---

# Account
(account_def)=
An account mirrors the state of the trading account of the underlying broker.
It contains available cash, open positions, open orders, available buying power
and executed trades.

The account doesn't contain closed orders and closed positions.

:::{tip}
If you want to track all orders during a run, use the {cl}`SignalOrderTracker` journal.
:::


Account is the only created and modified by the {cl}`Broker` and returned when the `sync()` method is invoked. 
It is also the object returned from the {cl}`run()` function.

```{code-cell} python
import roboquant as rq
account = rq.demo_run()
print(account)
```

There are several helper methods available that can help to further inspect the account after a run.

```{code-cell} python
account.orders_to_dataframe()
account.trades_to_dataframe()
account.pnl()
```