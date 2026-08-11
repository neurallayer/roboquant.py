---
kernelspec:
  name: python3
  display_name: Python 3
---

# Feature 

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

In *roboquant* there are three types of feature implementations included:

1. Those that calculate based on an event 
2. Those that calculate based on an account
3. Those that wrap other features, for example fill missing values


## Event-Based Features

| Feature | Description |
|---|---|
| `DayOfWeekFeature` | Day of week when the event took place |
| `DayOfMonthFeature` | Day of month when the event took place |
| `MonthOfYearFeature` | Month of year when the event took place |
| `TimeDifference` | Time difference between two events |
| `IndicatorFeature` | Build own indicators |
| `TrueRangeFeature` | Calculate True Range  |
| `PriceFeature` | Extract the prices for one or more assets |
| `BarFeature` | Extract the bar prices for one or more assets |
| `QuoteFeature` | Extract the quotes for one or more assets |
| `CacheFeature` | Cache other event feature |
| `VolumeFeature` | Extract the trading volume for one or more assets|


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
| `RandomFeature` | Feature of random values |
| `FeatureSet` | Combine other features into a new feature |
| `NormalizeFeature` | Z-score normalisation |
| `FillFeature` | Fill missing values |
| `FillWithConstantFeature` | Z-score normalisation |
| `ReturnFeature` | Calculate the return |
| `LongReturnsFeature` | Calculate the return over a longer period |
| `MaxReturnFeature` | Calculate the max return over certain period|
| `MinReturnFeature` | Calculate the min return over certain period |
| `SMAFeature` | Simple Mpving Average |


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

