---
kernelspec:
  name: python3
  display_name: Python 3
---

# Asset
(asset_def)=
The `Asset` class is the fundamental building block for representing financial instruments in *roboquant*.
Every tradable instrument — whether a stock, option, forex pair, or cryptocurrency —
is modelled as a specialized subclass of `Asset`.

## Overview
All asset types extend the common base class, `Asset`, which provides:

- A **symbol** (ticker)
- The currency the asset is denoted in
- Calculation of the contract value
- Serialization and de-serialization (used for example to store prices in databases)

Subclasses can add type-specific attributes and methods.

The combination of **asset class**, **symbol** and **currency** has to be unique.
If you deal with scenario's where this is not the case, a solution is to extend the symbol name. 

For example you want to differentiate between crypto-pairs on different exchanges,
add the exchange name to the symbol name:

```{code-cell} python
from roboquant import Crypto, USD

asset1 = Crypto("BTCUSD.KRAKEN", USD)
asset2 = Crypto("BTCUSD.BINANCE", USD)

assert asset1 != asset2
```

## Asset Classes
roboquant support the folowing asset types out of the box:

| Asset Type      | Subclass       | Typical Use Case                |
| :-------------- | :------------- | :------------------------------ |
| `STOCK `        | `Stock`        | Stocks, ETFs, ADRs              |
| `OPTION`        | `Option`       | Equity & index options          |
| `FOREX`         | `Forex`        | Currency pairs (EUR/USD)        |
| `CRYPTO`        | `Crypto`       | Crypto spot (BTC/USDT)          |


But you easily add your own asset classes by extending `Asset` class.

## Instantiation

```{code-cell} python
:tags: [hide-output]
import pprint
from roboquant import Stock, Crypto, Forex, Option, EUR

# Create a simple stock with the default currency (USD)
aapl = Stock("AAPL")
print(aapl)         

# Create a Banco Santander (Spain) stock in Euro
san = Stock("SAN", EUR)
print(san)

# use a static method to derive the currency
# from the symbol name
eurusd = Forex.from_symbol("EUR/USD")
print(eurusd)

btc = Crypto.from_symbol("BTC/USDT")
print(btc)

# Options using the OCC notation
option = Option("TSLA250228C00100000")
pprint.pp(option.decode_occ_symbol())
```
