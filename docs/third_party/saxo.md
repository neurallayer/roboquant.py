---
kernelspec:
  name: python3
  display_name: Python 3
---

# Saxo

## Overview
Saxo Bank is a Danish investment bank and a global online trading and investment platform. 

The `SaxoBroker` integrates roboquant with Saxo Bank's OpenAPI, enabling simulated order execution through a Saxo account.
It can be used standalone or passed into {cl}`run` for paper trading execution.

:::{note}
Right now *roboquant* only supports the API key authentication method.
That also limits the trading to using the Saxo sim environment and not their
live environment (requires OAUTH2). This might be added at a later stage.
:::

Saxo Bank support many assets and assets classes world wide. And currently *roboquant* pre-loads the Stocks, ETF's,
Funds and Forex totalling more than 25.000 assets.   

It is a simple and well documented API that you can checkout [here](https://www.developer.saxo/openapi/referencedocs).
---

## Prerequisite
In contrast to other broker integrations, the Saxo OpenAPI integrations only relies on the `requests` library and so
no additional libraries need to be loaded.

You'll need however create a temporary API key (24 hour valid) from your Saxo developer account.


## SaxoBroker — Paper Trading
The `SaxoBroker` allows you to execute orders against an Saxo sim account. Just 
make sure your temporary API key is still valid.

The broker can be used standalone like in the example below, or
passed into {cl}`run` for paper trading execution.

If you trade in different currencies, you need to register a `CurrencyConverter`.

```{code} python
import os
from dotenv import load_dotenv
from roboquant import Order
from roboquant.brokers.saxo import SaxoBroker

load_dotenv()

# Initialize broker
SAXO_TOKEN=os.environ["SAXO_TOKEN"]
broker = SaxoBroker(SAXO_TOKEN)

# Get account
account = broker.sync()
print(account)

# Place a market order
order = Order(Stock("TSLA"), Decimal(10))
broker.place_orders([order])

# Get updated account
account = broker.sync()
print(account)
```
