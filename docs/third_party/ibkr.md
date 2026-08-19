---
kernelspec:
  name: python3
  display_name: Python 3
---

# IBKR

## Overview
Interactive Brokers (IBKR) is a global electronic brokerage firm offering multi-asset trading, competitive pricing, and access to over 150 markets worldwide. It is one of the largest online brokers, widely used by professional traders, institutions, and algorithmic trading systems.

Interactive Brokers has several ways of connecting to it. Roboquant uses the Python IBind package. This is an unofficial Python API client library for the Interactive Brokers Client Portal Web API. It supports fully headless connectivity using OAuth 1.0a. So no more need to deploy Trader Workstation and connect through that.

:::{note}
One downside of the Client Portal Web API is that the API to get the unique asset identifiers (called `conid` in IBKR world) is rather flimsy.

So for example: placing an order for a US stock with the symbol name TSLA isn't as trivial as you might think. Consider to maintain a local mapping of the
assets you want to trade mapped to their counterpart IBKR `conid`.  
:::

## Broker
The IBind client expect a number of environment variables to be set. The whole process to get these variables is described in details at https://github.com/Voyz/ibind/wiki/OAuth-1.0a. The process includes uploading keys to IBKR. 

For sure IBKR doesn't make it as simple as just generating an API key 😉.

```properties
IBIND_USE_OAUTH=True
IBIND_USE_SESSION=True
IBIND_OAUTH1A_DH_PRIME=...
IBIND_OAUTH1A_CONSUMER_KEY=...
IBIND_OAUTH1A_ENCRYPTION_KEY_FP=...
IBIND_OAUTH1A_SIGNATURE_KEY_FP=..
IBIND_OAUTH1A_ACCESS_TOKEN=...
IBIND_OAUTH1A_ACCESS_TOKEN_SECRET=...
```

The upside is that the remaining Python code is straight forward. If you have more than one account, you can pass it as a parameter
to the IBKR broker.

```{code} python
from dotenv import load_dotenv
from roboquant.broker.ibkr import IBKRBroker

load_dotenv()

broker = IBKRBroker()
account = broker.sync()
```


