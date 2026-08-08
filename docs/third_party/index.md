---
kernelspec:
  name: python3
  display_name: Python 3
---

# Third Party
Roboquant has out of the box support for several third party data provides
and brokers.

1. Interactive Brokers
2. Alpaca
3. Crypto (via CCTX)

All the required dependencies can be added to your project in the following way

```bash
pip install roboquant[extra]
```

You have to import the feeds and brokers explicitely from their 
module.

```{code-cell} python
from roboquant.feeds.alpaca import AlpacaHistoricStockFeed
from roboquant.brokers.ibkr import IBKRBroker
```

