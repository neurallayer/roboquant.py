from typing import Type

from roboquant.common.asset import Asset, Crypto, Forex, Option, Stock
from roboquant.common.monetary import Currency
from roboquant.feeds.historicfeed import HistoricFeed


class PersistentFeed(HistoricFeed):
    """ABC class for deeds that require storing market data and assets.
    It contains utility methods to support asset (de-)serialization.
    """

    def __init__(self):
        self.__cache: dict[str, Asset] = {}
        self.__asset_classes: dict[str, Type[Asset]] = {}
        for assetClass in {Stock, Option, Crypto, Forex}:
            self.__asset_classes[assetClass.__name__] = assetClass

    def _serialize_asset(self, asset: Asset) -> str:
        """Serialize an asset to a string representation.

        Args:
            asset (Asset): The asset to serialize.

        Returns:
            The serialized string representation of the asset.
        """

        match asset:
            case Stock() | Crypto() | Forex() | Option():
                result = f"{asset.asset_class}{asset.symbol}{asset.currency}"
                return result
            case _:
                raise ValueError(f"unsupported asset type {type(asset)}")

    def _deserialize_to_asset(self, value: str) -> Asset:
        """Based on the provided string value, deserialize it to the corresponding asset.
        The asset class needs to be registered first using the `register_asset_class` method.

        Under the hood is uses caching to improve performance when repeatedly deserializing the same asset strings.

        Args:
            The serialized string representation of the asset.

        Returns:
            The deserialized asset.
        """
        asset = self.__cache.get(value)
        if not asset:
            asset_class, symbol, code = value.split("")
            asset = self.__asset_classes[asset_class](symbol, Currency(code))
            self.__cache[value] = asset
        return asset
