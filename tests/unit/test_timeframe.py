import unittest
from datetime import datetime, timedelta, timezone

from roboquant import Timeframe


class TestTimeframe(unittest.TestCase):

    def test_timeframe_creation(self):
        tf1 = Timeframe.fromisoformat("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00")
        tf2 = Timeframe(datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2021, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(tf1, tf2)

    def test_timeframe_inclusive(self):
        tf = Timeframe.next(days=2, inclusive=True)
        self.assertEqual("UTC", tf.start.tzname())
        self.assertEqual("UTC", tf.end.tzname())

        now = datetime.now(timezone.utc)
        self.assertTrue(now in tf)
        self.assertTrue(tf.start in tf)
        self.assertTrue(tf.end in tf)

    def test_timeframe_exclusive(self):
        tf = Timeframe.next(days=2, inclusive=False)
        now = datetime.now(timezone.utc)
        self.assertTrue(now in tf)
        self.assertTrue(tf.start in tf)
        self.assertFalse(tf.end in tf)

    def test_timeframe_duration(self):
        duration = timedelta(days=2)
        now = datetime.now(timezone.utc)
        tf = Timeframe(now, now + duration)
        self.assertEqual(duration, tf.duration)

    def test_timeframe_split(self):
        tf = Timeframe.fromisoformat("2000-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")
        tfs = tf.split(5)
        self.assertEqual(5, len(tfs))
        self.assertEqual(tf.start, tfs[0].start)
        self.assertEqual(tf.end, tfs[-1].end)

        tfs = tf.split(timedelta(days=365 * 3))
        self.assertEqual(7, len(tfs))
        self.assertEqual(tf.start, tfs[0].start)
        self.assertEqual(tfs[0].duration, timedelta(days=365 * 3))
        self.assertEqual(tf.end, tfs[-1].end)

    def test_timeframe_sample(self):
        tf = Timeframe.fromisoformat("2000-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")
        days_365 = timedelta(days=365)
        tfs = tf.sample(days_365, 10)
        self.assertEqual(10, len(tfs))
        for t in tfs:
            self.assertGreaterEqual(t.start, tf.start)
            self.assertGreaterEqual(tf.end, t.end)
            self.assertEqual(t.end - t.start, days_365)


if __name__ == "__main__":
    unittest.main()
