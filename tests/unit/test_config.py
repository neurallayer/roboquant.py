from configparser import NoOptionError
import unittest

import roboquant as rq


class TestConfig(unittest.TestCase):

    def test_config(self):
        config = rq.Config()
        with self.assertRaises(NoOptionError):
            config.get("unknown_key")

        with self.assertRaises(AssertionError):
            config = rq.Config("non_existing_file.conf")


if __name__ == "__main__":
    unittest.main()
