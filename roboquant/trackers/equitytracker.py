from datetime import datetime
from .tracker import Tracker


class EquityTracker(Tracker):
    """Tracks the time of an event and the equity at that moment.
    If multiple events happen at the same time, only the first one will be registered.
    """

    def __init__(self):
        self.timeline = []
        self.equity = []
        self.last = datetime.fromisoformat("1900-01-01T00:00:00+00:00")

    def log(self, event, account, ratings, orders):
        if event.time > self.last:
            self.timeline.append(event.time)
            self.equity.append(account.equity)
            self.last = event.time
