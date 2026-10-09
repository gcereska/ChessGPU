import os

import pytest

# Qt widgets render offscreen, so the tests need no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helpers import engine_path  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    """Path of the engine executable; skips the test if it isn't built."""
    path = engine_path()
    if path is None:
        pytest.skip("engine not built: build Release or set CHESSGPU_ENGINE")
    return path


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
