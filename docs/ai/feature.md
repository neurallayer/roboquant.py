---
kernelspec:
  name: python3
  display_name: Python 3
---

# Feature 
(feature_def)=
Feature classes in *roboquant* transform data into structured input features and target labels for AI/ML models.
They serve as the bridge between the time-series domain of financial data and the tabular world of machine learning algorithms.

The following code snippet shows the essence of the Feature abstract base class:

```python
class Feature(Generic[T]):
   
    @abstractmethod
    def calc(self, value: T) -> NDArray[np.float32]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

```
As can seen in the above snippet, a feature calculation always returns an Numpy array of the type float32.
In fact it is aways a 1-dimensional float32 array and missing values are represented as float "NaN" values in
that array. Every invocation should always return the same length array.

In *roboquant* there are three types of feature implementations included:

1. Features that calculate based on an event 
2. Features that calculate based on an account
3. Generic Features that wrap other features, for example fill missing values or don't rely 
   rely on input data at all.


## Event-Based Features

| Feature | Description |
|---|---|
| `DayOfWeekFeature` | Day of week when the event took place |
| `DayOfMonthFeature` | Day of month when the event took place |
| `MonthOfYearFeature` | Month of year when the event took place |
| `TimeDifference` | Time difference between two events |
| `IndicatorFeature` | Base class for own indicators |
| `TrueRangeFeature` | Calculate True Range  |
| `PriceFeature` | Extract the prices for one or more assets |
| `BarFeature` | Extract the bar prices for one or more assets |
| `QuoteFeature` | Extract the quotes for one or more assets |
| `CacheFeature` | Cache other event feature |
| `VolumeFeature` | Extract the trading volume for one or more assets|


Besides writing a full custom feature, you can also implement the 
Indicator feature.

```{code-cell} python
from roboquant.util.indicators import RSI
import roboquant as rq
from roboquant.common.asset import Asset
from roboquant.ai.features import IndicatorFeature
from roboquant.util.buffer import OHLCVBuffer

class RSIFeature(IndicatorFeature):
    """Example using TaLib to create a RSI feature"""

    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        close = ohlcv.close()
        return RSI(close, timeperiod=self.timeperiod - 1)
```


## Account-Based Features

| Feature | Description |
|---|---|
| `EquityFeature` | Total Equity |
| `UnrealizedPNLFeature` | Unrelaized Profit & Loss|


## Generic Features

| Feature | Description |
|---|---|
| `SlicedFeature` | Slice another feature |
| `FixedValueFeature` | Feature of fixed values |
| `RandomFeature` | Feature that generates random values |
| `FeatureSet` | Combine other features into a new feature |
| `NormalizeFeature` | Normalize data over a certain period |
| `FillFeature` | Fill in missing ("nan") values iwht last known value|
| `FillWithConstantFeature` | Fill missing values with a constant float value |
| `ReturnFeature` | Calculate the next step return |
| `LongReturnsFeature` | Calculate the return over a longer period |
| `MaxReturnFeature` | Calculate the maximum return over certain period|
| `MinReturnFeature` | Calculate the minimal return over certain period |
| `SMAFeature` | Calculate the Simple Moving Average over the result of another Feature  |


## Custom Features
For bespoke logic, subclass `Feature` and implement the `calc()` and `size()` method.
The following example shows how to develop a feature that calculates the spreads between
two different assets.

```{code-cell} python
from roboquant import Event, Asset
from roboquant.ai.features import Feature
import numpy as np

class SpreadFeature(Feature[Event]):
    """This feature calculates the spread between two different assets"""

    def __init__(self, asset_a: Asset, asset_b: Asset):
        self.asset_a = asset_a
        self.asset_b = asset_b

    def size(self):
        return 1

    def calc(self, event):
        px_a = event.get_price(self.asset_a)
        px_b = event.get_price(self.asset_b)
        if px_a is None or px_b is None:
            result = float("nan")
        else:
            result = px_a - px_b
        return np.array([result]) 
```

