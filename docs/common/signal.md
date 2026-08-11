---
kernelspec:
  name: python3
  display_name: Python 3
---

# Signal
The output of a `Strategy` is a list of `Signal` objects. Each signal contains three pieces of
information:

- **`asset`** — the asset the signal applies to.
- **`rating`** — a float, normally between -1.0 (strong sell) and 1.0 (strong buy). This range is
  however not enforced.
- **`type`** — a `SignalType` flag indicating how the signal may be used: `ENTRY` (open or increase a
  position), `EXIT` (close or reduce a position), or `ENTRY_EXIT` (both, the default).

There are several ways to create a signal:

| Constructor | rating | type | Use |
|---|---|---|---|
| `Signal.buy(asset)` | 1.0 | `ENTRY_EXIT` | Strong buy |
| `Signal.sell(asset)` | -1.0 | `ENTRY_EXIT` | Strong sell |
| `Signal.buy(asset, SignalType.ENTRY)` | 1.0 | `ENTRY` | Only open/increase a position |
| `Signal.sell(asset, SignalType.EXIT)` | -1.0 | `EXIT` | Only close/reduce a position |
| `Signal(asset, rating, type)` | custom | custom | Full control |

```{code-cell} python
from roboquant import Stock, Signal, SignalType

apple = Stock("AAPL")

buy = Signal.buy(apple)
print(buy)

sell = Signal.sell(apple, SignalType.EXIT)
print(sell)

custom = Signal(apple, 0.5, SignalType.ENTRY)
print(custom)
```

Some of the convenience properties on a signal: `is_buy`, `is_sell`, `is_entry`, and `is_exit`.

:::{note}
It is up the `Trader` to use the signal and its properties. For example, the 
`FlexTrader` uses the `rating`  to determine the order sizing and respects the 
`SignalType` when it comes to position sizing.
:::