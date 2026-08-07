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
Roboquant isn't designed to be a pure visual algo-trading tool. Charts are included to provide
insights into what is happening during a run but are not the basis for strategies.

There is no out-of-the-box support for candlestick charts and things like the drawing of support lines.
Although these can be added using third party packages like mplfinance, they are not the
focus area for roboquant.
:::