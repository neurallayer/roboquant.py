import tempfile
import unittest
from pathlib import Path

from roboquant.feeds import SQLFeed
from tests.common import get_feed, run_price_item_feed


class TestSQLFeed(unittest.TestCase):

    def test_sql_feed(self):
        path = tempfile.gettempdir()
        db_file = Path(path).joinpath("tmp.db")
        db_file.unlink(missing_ok=True)
        self.assertFalse(db_file.exists())

        feed = SQLFeed(db_file)

        origin_feed = get_feed()
        feed.record(origin_feed)
        self.assertTrue(db_file.exists())

        self.assertEqual(origin_feed.timeframe(), feed.timeframe())
        feed.create_index()

        self.assertEqual(set(origin_feed.assets()), set(feed.assets()))
        run_price_item_feed(feed, origin_feed.assets(), self)


if __name__ == "__main__":
    unittest.main()
