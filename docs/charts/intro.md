---
kernelspec:
  name: python3
  display_name: Python 3
---

# Intro
Charts in *roboquant* are all based on `matplotlib`. Either by directly invoking methods or via Pandas. They can be 
used in an interactive environment like Jupyter Notebooks, but also saved as a PDF.

Roboquant has a dark style for the charts, which can be enabled by calling the `set_dark_style()` function.
Besides the dark background, it also sets some other parameters for the charts, like the figure size, dpi and grids.

```{code-cell} python
import roboquant as rq

rq.set_dark_style()
```

:::{note}
Charting support is included to gain a better understanding what is happening during a run. 
However it is not the basis for strategies and therefor there is no out-of-the-box support
for things like candlestick charts and support lines.

However if desired, this can be easily added by using third party libraries like mpl finance.
:::