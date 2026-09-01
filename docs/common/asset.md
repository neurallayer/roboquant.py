---
kernelspec:
  name: python3
  display_name: Python 3
---

# Asset

(asset_def)=
## Overview
The {cl}`Asset` class is the fundamental building block for representing financial instruments in *roboquant*.
Every tradable instrument — whether a stock, option, forex pair, or cryptocurrency —
is modeled as a specialized subclass of {cl}`Asset`.

## Overview
All asset types extend the common base class, {cl}`Asset`, which provides:

- A **symbol** (ticker)
- The currency the asset is denoted in
- Calculation of the contract value
- Serialization and de-serialization (used for example to store prices in databases)

Subclasses can add type-specific attributes and methods.

The **symbol** has to be unique accross all assets and this is enforced. If you create a new asset
with the same symbol name as an existing asset but with other attributes, it will raise an exception.

If you deal with scenarios where there are conflicting symbol names, a solution is to extend the symbol name. For example,
you want to differentiate between crypto-pairs on different exchanges, add the exchange name to the symbol name:

```{code-cell} python
from roboquant import Crypto, USD

asset1 = Crypto("BTCUSD.KRAKEN", USD)
asset2 = Crypto("BTCUSD.BINANCE", USD)

assert asset1 != asset2
```

## Asset Classes
roboquant supports the following asset types out of the box:

| Asset Type      | Subclass       | Typical Use Case                |
| :-------------- | :------------- | :------------------------------ |
| `STOCK `        | `Stock`        | Stocks, ETFs, ADRs              |
| `OPTION`        | `Option`       | Equity & index options          |
| `FOREX`         | `Forex`        | Currency pairs (EUR/USD)        |
| `CRYPTO`        | `Crypto`       | Crypto spot (BTC/USDT)          |


You add your own asset classes by extending {cl}`Asset` class.

## Instantiation
Although most commonly the Assets are created by the `Feed` you use, you
can also create your own instances.

```{code-cell} python
:tags: [hide-output]
import pprint
from roboquant import Asset, Stock, Crypto, Forex, Option, EUR

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

## Inspection
It is possible to see which assets has been created so far in
your application.

```{code-cell} python
:tags: [hide-output]
# print all unique assets instantiated so far
pprint.pp(Asset.assets())

# print only the unique Stocks
pprint.pp(Stock.assets())
```