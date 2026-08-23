from typing import TypedDict, List

class NetPositionBase(TypedDict):
    AccountId: str
    Amount: int
    AmountLong: int
    AmountShort: int
    AssetType: str
    CanBeClosed: bool
    ClientId: str
    HasForceOpenPositions: bool
    IsMarketOpen: bool
    MarketState: str
    NonTradableReason: str
    NumberOfRelatedOrders: int
    OpenBondPoolFactor: float      # could be int, but float is safer for numeric values
    OpeningDirection: str
    OpenIpoOrdersCount: int
    OpenOrdersCount: int
    OpenTriggerOrdersCount: int
    PositionsAccount: str
    TradingStatus: str
    Uic: int

class NetPositionView(TypedDict):
    AverageOpenPrice: float
    AverageOpenPriceIncludingCosts: float
    CalculationReliability: str
    ConversionRateCurrent: float
    CurrentBondPoolFactor: float
    CurrentPrice: float            # may be int 0, but price fields are better as float
    CurrentPriceDelayMinutes: int
    CurrentPriceType: str
    Exposure: float                # 0 could be int, but float covers all
    ExposureCurrency: str
    ExposureInBaseCurrency: float
    InstrumentPriceDayPercentChange: float
    MarketValueOpen: float
    MarketValueOpenInBaseCurrency: float
    PositionCount: int
    PositionsAverageBuyPrice: float
    PositionsAverageSellPrice: float
    PositionsNotClosedCount: int
    ProfitLossCurrencyConversion: float
    ProfitLossOnTrade: float
    ProfitLossOnTradeInBaseCurrency: float
    Status: str
    TradeCostsTotal: float
    TradeCostsTotalInBaseCurrency: float
    UnderlyingCurrentPrice: float

class NetPositionItem(TypedDict):
    NetPositionBase: NetPositionBase
    NetPositionId: str
    NetPositionView: NetPositionView

class NetPositionsResponse(TypedDict):
    __count: int
    Data: List[NetPositionItem]


class Duration(TypedDict):
    DurationType: str  # e.g., "GoodTillCancel", "DayOrder"

class Exchange(TypedDict):
    Description: str
    ExchangeId: str
    IsOpen: bool
    TimeZoneId: str

# For RelatedOpenOrders, if it's an array of objects, define a minimal type or use List[dict]
# Since it's empty in the example, we'll use List[dict] for flexibility.
RelatedOpenOrder = dict  # or define a more specific type if needed

class OpenOrderItem(TypedDict):
    AccountId: str
    AccountKey: str
    AdviceNote: str
    Amount: int
    AssetType: str
    BuySell: str
    CalculationReliability: str
    ClientId: str
    ClientKey: str
    ClientName: str
    ClientNote: str
    CorrelationKey: str
    CurrentPrice: float   # although 0 is int, price fields often float
    CurrentPriceDelayMinutes: int
    CurrentPriceType: str
    Duration: Duration
    Exchange: Exchange
    IpoSubscriptionFee: float
    IsExtendedHoursEnabled: bool
    IsForceOpen: bool
    IsMarketOpen: bool
    MarketPrice: float
    MarketState: str
    NonTradableReason: str
    OpenOrderType: str
    OrderAmountType: str
    OrderId: str
    OrderRelation: str
    OrderTime: str  # ISO datetime string
    Price: float    # order price, could be decimal
    RelatedOpenOrders: List[RelatedOpenOrder]
    Status: str
    TradingStatus: str
    Uic: int

class OpenOrdersResponse(TypedDict):
    __count: int
    Data: List[OpenOrderItem]
