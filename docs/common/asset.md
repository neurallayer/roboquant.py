---
kernelspec:
  name: python3
  display_name: Python 3
---

# Asset
(asset_def)=
The `Asset` class is the fundamental building block for representing financial instruments in *roboquant*. Every tradable instrument — whether a stock, option, forex pair, or cryptocurrency — is modelled as a specialised subclasses of `Asset`.

## Overview

All asset types share a common base class, `Asset`, which provides:

- A unique **symbol** (ticker)
- The currency the asset is denoted in
- Calculation of the contract value
- Serialization and de-serialization (used for example to store prices in databases)

Subclasses can add type-specific attributes and methods.


## Asset Classes

roboquant support the folowing asset types out of the box:

| Asset Type      | Subclass       | Typical Use Case                |
| :-------------- | :------------- | :------------------------------ |
| `STOCK `        | `Stock`        | Stocks, ETFs, ADRs              |
| `OPTION`        | `Option`       | Equity & index options          |
| `FOREX`         | `Forex`        | Currency pairs (EUR/USD)        |
| `CRYPTO`        | `Crypto`       | Crypto spot (BTC/USDT)          |



```{code-cell} python
from roboquant import Stock, Crypto, Forex, Option

# Create a simple stock with default currency USD
aapl = Stock("AAPL")
print(aapl)         

# use a static method
eurusd = Forex.from_symbol("EUR/USD")
print(eurusd)

btc = Crypto.from_symbol("BTC/USDT")
print(btc)

option = Option("TSLA250228C00100000")
print(option.decode_occ_symbol())
```
