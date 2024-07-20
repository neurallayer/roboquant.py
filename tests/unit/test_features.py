import unittest

import numpy as np
from roboquant.account import Account
from roboquant.event import Event

from roboquant.ml.features import (
    CacheFeature,
    CombinedFeature,
    NormalizeFeature,
    PriceFeature,
    SMAFeature,
    ReturnsFeature,
    VolumeFeature,
    FixedValueFeature,
    DayOfWeekFeature,
)
from tests.common import get_feed


class TestFeatures(unittest.TestCase):

    def test_all_features(self):
        feed = get_feed()
        symbols = list(feed.assets())
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        feature = CombinedFeature(
            PriceFeature(symbol1, price_type="CLOSE"),
            PriceFeature(symbol1, price_type="OPEN"),
            SMAFeature(FixedValueFeature(np.ones((3,))), 8),
            SMAFeature(PriceFeature(symbol2, price_type="CLOSE"), 10),
            ReturnsFeature(PriceFeature(symbol1, price_type="OPEN")),
            VolumeFeature(symbol2),
            DayOfWeekFeature(),
        )
        account = Account()
        channel = feed.play_background()
        while evt := channel.get():
            result = feature.calc(evt, account)
            self.assertTrue(len(result) == feature.size())

    def test_cache(self):
        feed = get_feed()

        feature = CacheFeature(
            PriceFeature(*feed.assets()),
        )

        account = Account()
        channel = feed.play_background()
        while evt := channel.get():
            result1 = feature.calc(evt, account).sum()
            result2 = feature.calc(evt, account).sum()
            if not np.isnan(result1):
                self.assertEqual(result1, result2)

    def test_slice(self):
        f = FixedValueFeature([1, 2, 3, 4, 5, 6, 7, 8])[1:6:2]
        self.assertEqual(3, f.size())

    def test_normalize(self):
        feed = get_feed()

        feature = CombinedFeature(
            PriceFeature(*feed.assets()).returns(),
        )

        norm_feature = NormalizeFeature(feature, 10)
        account = Account()
        channel = feed.play_background()
        while evt := channel.get():
            result = norm_feature.calc(evt, account)
            self.assertTrue(len(result) == feature.size())

    def test_core_feature(self):
        account = Account()
        f = FixedValueFeature(np.ones((10,)))[2:5]
        values = f.calc(Event.empty(), account)
        self.assertEqual(3, len(values))

        f = FixedValueFeature(
            np.ones(
                10,
            )
        ).returns()
        f.calc(Event.empty(), account)
        values = f.calc(Event.empty(), account)
        self.assertEqual(0, values[0])


if __name__ == "__main__":
    unittest.main()
