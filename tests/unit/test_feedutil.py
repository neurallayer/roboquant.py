from decimal import Decimal
from roboquant.common.monetary import USD
from roboquant.common.asset import Forex
from roboquant.common.asset import Stock
from roboquant.feeds.util import AssetSerializer
import unittest
from datetime import timedelta

from roboquant.feeds.randomwalk import RandomWalk
from roboquant.feeds.util import BarAggregatorFeed, TimeGroupingFeed
from tests.common import run_price_item_feed


class TestFeedUtil(unittest.TestCase):

    def test_bar_aggregator_feed(self):
        feed = RandomWalk(5, 200, price_type="trade", frequency=timedelta(seconds=1))
        candle_feed = BarAggregatorFeed(feed, timedelta(seconds=60), price_type="trade")
        run_price_item_feed(candle_feed, feed.assets(), self)

    def test_time_grouping_feed(self):
        feed = RandomWalk(5, 200, price_type="trade", frequency=timedelta(seconds=1))
        grouped_feed = TimeGroupingFeed(feed, timeout=10.0)
        run_price_item_feed(grouped_feed, feed.assets(), self)

    def test_serializer(self):
        ser = AssetSerializer()
        abc = Stock("ABC")
        abc_ser = ser._serialize_asset(abc)
        self.assertEqual(abc_ser, "Stock\x1fABC\x1fUSD")
        abc2 = ser._deserialize_to_asset(abc_ser)
        self.assertEqual(abc, abc2)

        f1 = Forex("BTCUSD", USD, None, Decimal(10_000))
        f1_ser = ser._serialize_asset(f1)
        f2 = ser._deserialize_to_asset(f1_ser)
        self.assertEqual(f1, f2)


if __name__ == "__main__":
    unittest.main()
