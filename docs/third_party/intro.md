---
kernelspec:
  name: python3
  display_name: Python 3
---

# Third Party
Roboquant has out of the box support for several third party data provides
and brokers.

The required dependencies can be added to your project in the following way

```bash
pip install roboquant[extra]
```


```{code-cell} python
from roboquant.feeds.alpaca import AlpacaHistoricStockFeed
```

