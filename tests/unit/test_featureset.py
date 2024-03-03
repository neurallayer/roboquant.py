import unittest

import numpy as np

from roboquant.strategies.features import (
    FeatureSet,
    PriceFeature,
    SMAFeature,
    ReturnsFeature,
    VolumeFeature,
    FixedValueFeature,
    DayOfWeekFeature,
)
from tests.common import get_feed


class TestFeatureSet(unittest.TestCase):

    def test_feature_set(self):
        feed = get_feed()
        symbols = feed.symbols
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        warmup = 20
        history_size = 100
        fs = FeatureSet(history_size, warmup=warmup)
        fs.add(PriceFeature(symbol1, "CLOSE"))
        fs.add(PriceFeature(symbol1, "OPEN"))
        fs.add(SMAFeature(FixedValueFeature("DUMMY", np.ones((3,))), 8))
        fs.add(SMAFeature(PriceFeature(symbol2, "CLOSE"), 10))
        fs.add(ReturnsFeature(PriceFeature(symbol1, "OPEN")))
        fs.add(VolumeFeature(symbol2))
        fs.add(DayOfWeekFeature())

        channel = feed.play_background()
        cnt = 0
        while evt := channel.get():
            fs.process(evt)
            cnt += 1

        self.assertEqual(history_size, len(fs._buffer))

    def test_fs_data(self):
        feed = get_feed()
        symbol1 = feed.symbols[0]

        warmup = 20
        history_size = 300
        fs = FeatureSet(history_size, warmup=warmup)
        fs.add(PriceFeature(symbol1, "CLOSE"))
        fs.add(PriceFeature(symbol1, "OPEN"))
        fs.add(VolumeFeature(symbol1))

        channel = feed.play_background()
        cnt = 0
        while evt := channel.get():
            fs.process(evt)
            cnt += 1

        data = fs.data()
        self.assertTrue(~np.isnan(data).all())

        corr = np.corrcoef(data, rowvar=False)
        self.assertTrue(~np.isnan(corr).all())

        diff = np.diff(data, axis=0)
        self.assertEqual(history_size - 1, diff.shape[0])

        self.assertTrue(fs._norm is None)
        fs.calc_norm()
        self.assertEqual((3, 2), fs._norm.shape)

        norm_data = fs.normalize()
        self.assertEqual(data.shape, norm_data.shape)
        self.assertAlmostEqual(0.0, norm_data.mean(), places=2)


if __name__ == "__main__":
    unittest.main()
