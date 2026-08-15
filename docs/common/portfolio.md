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


# Portfolio

(portfolio_def)=
## Overview
The portfolio holds the open positions and is one of the attributes of the {cl}`Account` object.

It behaves very much like a `dict[Asset, Position]`. So there is at most only one `Position` 
per asset. If a Position is zero, it will be removed from the Portfolio.

Only the {cl}`Broker` modifies the portfolio and its positions.

```{code-cell} python
:tags: [hide-output]
portfolio = account.portfolio

asset = rq.Stock("JPM")
pprint.pp(portfolio[asset])

print(f"unrealized pnl {portfolio.unrealized_pnl():_.2f}")

print(f"market value {portfolio.mkt_value():_.2f}")
```

## Position
(position_def)=
A position is the net quantity of an {cl}`Asset` currently held, representing its market exposure and risk at any given moment.
Every time a trade is executed, the Position for that Asset is updated.

The Position class is immutable and contains 3 attributes:

- contract size (combined result of all trades)
- average price paid (combined result of all trades)
- latest market price (updated at the end of each step)

:::{note} Avg price logic
The average price is the average of the execution prices. 
For example:
- current position => size is +10 and avg price is 100.00
- new trade => size is +40 and execution price is110.00
  
{math}`new\ avg\ price = \frac{(10*100.00) + (40*110.00)}{10 + 40} = 108.00`
:::

