import os
import os.path
from pathlib import Path
from configparser import ConfigParser


class Config:
    """Access to a roboquant configuration file.
    This allows sharing the same property file between both the Python and Kotlin
    version of roboquant.

    If no path is provided, the Config will search for "~/.roboquant/.env"
    """

    def __init__(self, path: Path | str | None = None):
        self.config = ConfigParser()
        if path:
            assert Path(path).exists(), "invalid path"
        path = path or os.path.expanduser("~/.roboquant/.env")
        config_string = "[default]\n"
        if Path(path).exists():
            with open(path, "r", encoding="utf8") as f:
                config_string += f.read()
        self.config.read_string(config_string)

    def get(self, key: str) -> str:
        """Get the value for the given key. It will first look in the environment variables and then in the config file"""
        for key2, value in os.environ.items():
            final_key = key2.lower().replace("_", ".")
            if final_key == key:
                return value
        return self.config.get("default", key)
