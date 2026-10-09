"""Helpers for tests that run the Qt event loop."""

from PySide6.QtCore import QElapsedTimer
from PySide6.QtTest import QTest


def wait_until(predicate, timeout_ms=10_000, step_ms=5):
    """Processes events until predicate() is true; False on timeout."""
    timer = QElapsedTimer()
    timer.start()
    while not predicate():
        if timer.elapsed() > timeout_ms:
            return False
        QTest.qWait(step_ms)
    return True


class Recorder:
    """Records every emission of a signal as a tuple of its arguments."""

    def __init__(self, signal):
        self.calls = []
        signal.connect(lambda *args: self.calls.append(args))

    def __len__(self):
        return len(self.calls)
