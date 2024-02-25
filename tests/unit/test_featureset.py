import unittest
from roboquant.strategies.featureset import (
    FeatureSet,
    PriceFeature,
    SMAFeature,
    ReturnsFeature,
    VolumeFeature,
    FixedValueFeature,
    DayOfWeekFeature,
)
from roboquant.feeds import EventChannel, feedutil
from tests.common import get_feed
import numpy as np


class TestFeatureSet(unittest.TestCase):

    def test_featureset(self):
        feed = get_feed()
        symbols = feed.symbols
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        warmup = 20
        fs = FeatureSet(10_000, warmup=warmup)
        fs.add(PriceFeature(symbol1, "CLOSE"))
        fs.add(PriceFeature(symbol1, "OPEN"))
        fs.add(SMAFeature(FixedValueFeature("DUMMY", np.zeros((3,))), 8))
        fs.add(SMAFeature(PriceFeature(symbol2, "CLOSE"), 10))
        fs.add(ReturnsFeature(PriceFeature(symbol1, "OPEN")))
        fs.add(VolumeFeature(symbol2))
        fs.add(FixedValueFeature("DUMMY", np.zeros((3,))))
        fs.add(DayOfWeekFeature())

        channel = EventChannel()
        feedutil.play_background(feed, channel)

        cnt = 0
        while evt := channel.get():
            fs.process(evt)
            cnt += 1

        self.assertEqual(cnt - warmup, len(fs._data))


if __name__ == "__main__":
    unittest.main()
