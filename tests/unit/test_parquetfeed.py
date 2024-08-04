import tempfile
import unittest
from pathlib import Path

from roboquant.feeds.parquet import ParquetFeed
from tests.common import get_feed, run_price_item_feed


class TestParquetFeed(unittest.TestCase):

    def test_parquet_feed(self):
        path = tempfile.gettempdir()
        db_file = Path(path).joinpath("tmp.parquet")
        db_file.unlink(missing_ok=True)
        self.assertFalse(db_file.exists())

        feed = ParquetFeed(db_file)
        self.assertFalse(feed.exists())

        origin_feed = get_feed()
        feed.record(origin_feed)
        self.assertTrue(db_file.exists())

        self.assertEqual(set(origin_feed.assets()), set(feed.assets()))
        self.assertEqual(origin_feed.timeframe(), feed.timeframe())

        run_price_item_feed(feed, origin_feed.assets(), self)
        run_price_item_feed(feed, origin_feed.assets(), self, timeframe=feed.timeframe())
        db_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
