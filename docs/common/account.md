---
kernelspec:
  name: python3
  display_name: Python 3
---

# Account

(account_def)=
## Overview
An account mirrors the state of the trading account of the underlying broker.
It contains available cash, open positions, open orders, available buying power
and executed trades.

The account doesn't contain closed orders and closed positions.

:::{tip}
If you want to track all orders or singals during a run,
use the {cl}`TrackerJournal`.
:::

Account is immutable and created by the {cl}`Broker` during the `sync()` method. 
It is also the object that is returned from the {cl}`run()` function.

```{code-cell} python
import roboquant as rq
account = rq.demo_run()
print(account)
```

## Buying Power
The buying power indicates how much of the account can still be used to place
new orders. It is the amount of cash that is available for trading, potentially
increased by any margin that the broker allows.

The available cash is the amount of money currently not invested, minus any
amount reserved by open orders. When margin is enabled, the buying power can be
a multiple of the available cash.

```{tip}
A negative buying power typically means the account is using (too much) margin
and may be subject to a margin call.
```

The following example shows the available cash and buying power of the account.

```{code-cell} python
print("available cash:", account.cash)
print("buying power:", account.buying_power)
```


## Positions
One of the most important information in the account are the open positions.

A position represents the current exposure of the account to a single asset. 

Each {cl}`Position` contains:

- the **symbol** of the asset,
- the **size** (number of shares or units), and
- the **average open price** and resulting **value**.
- the last known **market price**.

The size can be positive (long) or negative (short). A position whose size is zero
is considered closed and is not included in the account anymore.

(positions_hedging_netting)=
### Hedging versus netting
When multiple orders are executed for the same asset, the broker has to decide how
to combine them into positions. Roboquant supports two strategies:

**Netting**
: Only a single position per asset exists. A new order in the opposite direction
  first reduces the existing position, and only the remaining quantity opens a new
  position in that direction (potentially flipping long to short and vice versa).

**Hedging**
: Multiple positions per asset can exist at the same time, allowing a long and a
  short position in the same asset to coexist. Opposite orders do not automatically
  reduce each other. For positions to close, you typically need to refer to the position
  when placing the order. 

```{tip}
With netting the exposure to an asset is the net sum of all its trades, while with
hedging the gross long and short exposure can be tracked separately.
```

The following example shows all open positions of the account as a Pandas dataframe, one
row per position.

```{code-cell} python
account.positions_to_dataframe()
```

## Trades
Trades are particular useful after a back test to see how P&L is distributed among
assets and what type of trades resulted in winers or losers.

```{code-cell} python
print(account.realized_pnl())
account.trades_to_dataframe().sort_values(by='pnl').head()
```