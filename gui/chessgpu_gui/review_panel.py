"""The game review's dialog (engine, time per position, annotation) and its
panel: each side's accuracy, average centipawn loss and mistakes, the
moves judged (clicking one shows it), and the progress while it runs."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from .game import GameRecord, Node
from .panels import figurine_san
from .profiles import EngineProfile
from .review import KIND_TITLES, Judgement, SideSummary
from .rules import WHITE, IllegalMoveError
from .theme import current_theme

# (label, movetime ms, depth): how long the engine looks at each position.
REVIEW_SPEEDS = (("Quick: 0.3 s per position", 300, None), ("Normal: 1 s per position", 1000, None),
                 ("Deep: 3 s per position", 3000, None), ("Depth 12", None, 12), ("Depth 16", None, 16))
DEFAULT_SPEED = 1


def kind_color(kind: str) -> str:
    t = current_theme()
    return {"inaccuracy": "#d6b94a", "mistake": t.warning, "blunder": t.danger}[kind]


class ReviewDialog(QDialog):
    """Asks which engine, how long per position, and whether to annotate."""

    def __init__(self, profiles: list[EngineProfile], engine: str, positions: int, last: dict | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Game review")
        last = last or {}
        self.engine = QComboBox()
        self.engine.addItems([p.name for p in profiles])
        if engine in [p.name for p in profiles]:
            self.engine.setCurrentText(engine)
        self.speed = QComboBox()
        self.speed.addItems([s[0] for s in REVIEW_SPEEDS])
        try:
            self.speed.setCurrentIndex(max(0, min(len(REVIEW_SPEEDS) - 1, int(last.get("speed", DEFAULT_SPEED)))))
        except (TypeError, ValueError):
            self.speed.setCurrentIndex(DEFAULT_SPEED)
        self.annotate = QCheckBox("Annotate the game: marks, comments and the engine's lines")
        self.annotate.setChecked(str(last.get("annotate", "true")) != "false")
        self.estimate = QLabel()
        self.positions = positions
        self.speed.currentIndexChanged.connect(self._update_estimate)
        form = QFormLayout()
        form.addRow("Engine", self.engine)
        form.addRow("Time", self.speed)
        form.addRow("", self.annotate)
        form.addRow("", self.estimate)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Review")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update_estimate()

    def choice(self) -> tuple[str, int | None, int | None, bool]:
        """(engine name, movetime ms, depth, annotate)."""
        _label, movetime, depth = REVIEW_SPEEDS[self.speed.currentIndex()]
        return self.engine.currentText(), movetime, depth, self.annotate.isChecked()

    def settings(self) -> dict:
        return {"speed": self.speed.currentIndex(), "annotate": "true" if self.annotate.isChecked() else "false"}

    def _update_estimate(self) -> None:
        _label, movetime, _depth = REVIEW_SPEEDS[self.speed.currentIndex()]
        if movetime:
            seconds = self.positions * movetime / 1000
            self.estimate.setText(f"{self.positions} positions: about {seconds / 60:.1f} min" if seconds >= 90 else
                                  f"{self.positions} positions: about {seconds:.0f} s")
        else:
            self.estimate.setText(f"{self.positions} positions")


class ReviewPanel(QWidget):
    """The review's results and progress."""

    node_selected = Signal(object)  # Node
    cancel_requested = Signal()
    review_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("reviewPanel")
        self.status = QLabel("Review a game to see each side's accuracy and mistakes.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_requested.emit)
        self.start = QPushButton("Review game…")
        self.start.clicked.connect(self.review_requested.emit)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel)
        self.table = QGridLayout()
        self.table.setHorizontalSpacing(14)
        self.cells: dict[tuple[int, str], QLabel] = {}
        for col, title in enumerate(("", "Accuracy", "ACPL", "?!", "?", "??")):
            head = QLabel(title)
            head.setObjectName("fieldName")
            head.setAlignment(Qt.AlignmentFlag.AlignCenter if col else Qt.AlignmentFlag.AlignLeft)
            self.table.addWidget(head, 0, col)
        for row, color in ((1, WHITE), (2, 1 - WHITE)):
            for col, key in enumerate(("name", "accuracy", "acpl", "inaccuracies", "mistakes", "blunders")):
                cell = QLabel("–")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter if col else Qt.AlignmentFlag.AlignLeft)
                if key == "accuracy":
                    cell.setObjectName("accuracy")
                self.table.addWidget(cell, row, col)
                self.cells[(color, key)] = cell
        self.moves = QListWidget()
        self.moves.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.moves.itemClicked.connect(self._clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.status)
        layout.addLayout(progress_row)
        layout.addLayout(self.table)
        layout.addWidget(QLabel("Moves to look at:"))
        layout.addWidget(self.moves, 1)
        layout.addWidget(self.start, 0, Qt.AlignmentFlag.AlignRight)
        self.set_running(False)

    def set_running(self, running: bool, done: int = 0, total: int = 0) -> None:
        self.progress.setVisible(running)
        self.cancel.setVisible(running)
        self.start.setEnabled(not running)
        if running:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(done)
            self.progress.setFormat(f"Reviewing: {done} of {total} positions")

    def show_results(self, record: GameRecord, judgements: list[Judgement], sides: dict[int, SideSummary],
                     figurines: bool = False) -> None:
        names = {WHITE: record.tags.get("White", "White") or "White", 1 - WHITE: record.tags.get("Black", "Black") or
                 "Black"}
        for color, s in sides.items():
            self.cells[(color, "name")].setText(names[color])
            self.cells[(color, "accuracy")].setText(f"{s.accuracy:.0f}%" if s.moves else "–")
            self.cells[(color, "acpl")].setText(f"{s.acpl:.0f}" if s.moves else "–")
            for key, value in (("inaccuracies", s.inaccuracies), ("mistakes", s.mistakes), ("blunders", s.blunders)):
                self.cells[(color, key)].setText(str(value) if s.moves else "–")
        self.moves.clear()
        for j in judgements:
            if j.kind is None:
                continue
            item = QListWidgetItem(self.describe(j, figurines))
            item.setForeground(QColor(kind_color(j.kind)))
            item.setData(Qt.ItemDataRole.UserRole, j.node)
            self.moves.addItem(item)
        reviewed = sum(s.moves for s in sides.values())
        self.status.setText(f"{reviewed} moves reviewed." if reviewed else "Nothing reviewed yet.")

    @staticmethod
    def describe(j: Judgement, figurines: bool = False) -> str:
        before = j.node.parent.position
        number = f"{before.fullmove}." if before.turn == WHITE else f"{before.fullmove}..."
        san = figurine_san(j.node.san) if figurines else j.node.san
        text = f"{number} {san}  {KIND_TITLES[j.kind]}  (−{100 * j.loss:.0f}%)"
        if j.best:
            try:
                best = before.san(before.parse_uci(j.best))
                text += f", {figurine_san(best) if figurines else best} was best"
            except IllegalMoveError:
                pass
        return text

    def clear(self) -> None:
        for (_color, key), cell in self.cells.items():
            cell.setText("–")
        self.moves.clear()
        self.status.setText("Review a game to see each side's accuracy and mistakes.")

    def _clicked(self, item: QListWidgetItem) -> None:
        node = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(node, Node):
            self.node_selected.emit(node)
