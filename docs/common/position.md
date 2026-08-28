---
kernelspec:
  name: python3
  display_name: Python 3
---


```{code-cell} python
:tags: [remove-input]
import roboquant as rq
import pprint
account = rq.demo_run()
```


# Position

(position_def)=
## Overview

A position is the quantity of an {cl}`Asset` currently held, representing its market exposure and risk at any given moment.
Every time a trade is executed, the Position for that Asset is updated or a new position is created.

The Position class is immutable and contains 4 attributes:

- asset of the position
- contract size (combined result of all trades)
- average price paid (combined result of all trades)
- latest market price (updated at the end of each step)

:::{note} Netting positioning logic
The average price is the average of the execution prices. 
For example:
- current position ⇒ size is +10 and avg price is 100.00
- new trade ⇒ size is +40 and execution price is110.00
  
{math}`new\ avg\ price = \frac{(10*100.00) + (40*110.00)}{10 + 40} = 108.00`
:::

```{code-cell} python
:tags: [hide-output]

for position in account.positions:
  print(position.asset, position.size, position.avg_price, position.mkt_price)

print(f"unrealized pnl {account.unrealized_pnl():_.2f}")
print(f"market value {account.mkt_value():_.2f}")
```
