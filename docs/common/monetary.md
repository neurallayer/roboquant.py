---
kernelspec:
  name: python3
  display_name: Python 3
---

# Monetary

## Overview
Roboquant comes with several monetary classes that makes trading with 
multi-currency situations a lot easier.

## Currency
(currency_def)=
A `Currency` represents a single currency (e.g. USD, EUR, JPY). Predefined currency
instances are available as module-level constants and can be used directly.
Currencies are not limited to fiat currencies and can be used for crypto currencies as well.

Currencies are callable and then create an {cl}`Amount`. Or you can use the '@' operator to
achieve the same.


```{code-cell} python
from roboquant.common.monetary import USD, EUR, JPY, Currency

# Create a custom currency
DOGE = Currency("DOGE")

amount = DOGE(1000.0)
print(amount)

amount = 1000.0@DOGE
print(amount)
```

## Amount
(amount_def)=

An amount hold a monetary value of a single currency. Amounts are immutable.
Internally amount values are stored as floats since that is precise enough for
trading and it doesn't has the overhead of some other types.

Roboquant is not a ledger and for trading it doesn't need to be.

```{code-cell} python
from roboquant.common.monetary import Amount

amt1 = 20@USD
amt2 = Amount(USD, 20)
amt3 = USD(20)

assert amt1 == amt2 == amt3 

print(amt3)

# Amounts can be formatted like floats
print(f"{amt3:.0f}")
```


## Wallet
(wallet_def)=
A wallet can contain monetary values of different currencies. A wallet is mutable and 
acts very much like a `dict[Currency, float]` 

```{code-cell} python
from roboquant.common.monetary import Wallet, JPY

wallet = Wallet(200@USD, 10@EUR)
wallet += 20@USD + 30@EUR - 10@USD + 1000@JPY
wallet.deposit(100@EUR)
wallet.withdraw(200@JPY)

print(wallet)

# Wallets can be formatted like floats
print(f"{wallet:.0f}")

# Access wallet amounts by currency
print(wallet[USD], wallet[EUR], wallet[JPY])
```

## CurrencyConverter
(currencyconverter_def)=
If trading in more than one currency, a conversion is required 
between amounts in the different currencies. 

Roboquant comes out of the box with a conversion using the exchange 
rates as published by the ECB. 

```{code-cell} python
from roboquant.common.monetary import ECBConversion
from datetime import datetime

ECBConversion().register()

t1 = datetime.fromisoformat("2010-01-01")
print(USD, wallet.convert_to(USD,t1))
print(EUR, wallet.convert_to(EUR,t1))
```

