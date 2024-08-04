import tempfile
import unittest
from pathlib import Path

from roboquant.feeds.avro import AvroFeed
from tests.common import get_feed, run_price_item_feed


class TestAvroFeed(unittest.TestCase):

    def test_avro_feed(self):
        path = tempfile.gettempdir()
        db_file = Path(path).joinpath("tmp.parquet")
        db_file.unlink(missing_ok=True)
        self.assertFalse(db_file.exists())

        feed = AvroFeed(db_file)
        self.assertFalse(feed.exists())

        origin_feed = get_feed()
        feed.record(origin_feed)
        self.assertTrue(db_file.exists())
        # print(feed.index())

        run_price_item_feed(feed, origin_feed.assets(), self)
        db_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
