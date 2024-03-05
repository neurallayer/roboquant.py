import unittest

import numpy as np
from roboquant.event import Event

from roboquant.strategies.features import (
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
        symbols = feed.symbols
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        fs = [
            PriceFeature(symbol1, "CLOSE"),
            PriceFeature(symbol1, "OPEN"),
            SMAFeature(FixedValueFeature(np.ones((3,))), 8),
            SMAFeature(PriceFeature(symbol2, "CLOSE"), 10),
            ReturnsFeature(PriceFeature(symbol1, "OPEN")),
            VolumeFeature(symbol2), DayOfWeekFeature()
        ]

        channel = feed.play_background()
        while evt := channel.get():
            for feature in fs:
                result = feature.calc(evt)
                self.assertTrue(len(result) > 0)

    def test_core_feature(self):
        f = FixedValueFeature(np.ones(10,))[2:5]
        values = f.calc(Event.empty())
        self.assertEqual(3, len(values))

        f = FixedValueFeature(np.ones(10,)).returns()
        values = f.calc(Event.empty())
        values = f.calc(Event.empty())
        self.assertEqual(0, values[0])


if __name__ == "__main__":
    unittest.main()
