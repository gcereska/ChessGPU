"""Starts the GUI."""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    QApplication.setOrganizationName("ChessGPU")
    QApplication.setApplicationName("ChessGPU GUI")
    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.show()
    return app.exec()
