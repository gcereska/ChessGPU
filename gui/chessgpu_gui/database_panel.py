"""The database panel (the open PGN files' games: search, filters, the
position on the board) and the opening explorer (the moves played from the
position on the board, with how the games went)."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QStyledItemDelegate, QTableView, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .database import DbGame, GameDatabase
from .panels import figurine_san
from .rules import Move, Position
from .theme import current_theme

COLUMNS = (("#", None), ("White", "White"), ("Black", "Black"), ("Result", "Result"), ("Date", "Date"),
           ("Event", "Event"), ("ECO", "ECO"))


class GamesModel(QAbstractTableModel):
    """The games shown (a search's result), sortable by any column."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.games: list[DbGame] = []

    def set_games(self, games: list[DbGame]) -> None:
        self.beginResetModel()
        self.games = list(games)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.games)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def value(self, game: DbGame, column: int):
        tag = COLUMNS[column][1]
        return game.number if tag is None else (game.result if tag == "Result" else game.tag(tag))

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        game = self.games[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.value(game, index.column()))
        if role == Qt.ItemDataRole.UserRole:
            return game.number
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section][0]
        return None

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        self.layoutAboutToBeChanged.emit()
        self.games.sort(key=lambda g: (self.value(g, column) if column else g.number),
                        reverse=order == Qt.SortOrder.DescendingOrder)
        self.layoutChanged.emit()


class DatabasePanel(QWidget):
    """The open database's games, searched; double-clicking one opens it."""

    open_requested = Signal()
    add_requested = Signal()
    close_requested = Signal()
    game_requested = Signal(int)  # game number
    search_changed = Signal()  # the filters changed (the window searches again)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("databasePanel")
        self.status = QLabel("No database. Open a PGN file of games.")
        self.status.setWordWrap(True)
        self.open = QPushButton("Open…")
        self.add = QPushButton("Add file…")
        self.close = QPushButton("Close")
        for b, s in ((self.open, self.open_requested), (self.add, self.add_requested), (self.close, self.close_requested)):
            b.clicked.connect(s.emit)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.text = QLineEdit()
        self.text.setPlaceholderText("Player, event, site, opening…")
        self.text.setClearButtonEnabled(True)
        self.result = QComboBox()
        self.result.addItems(["Any result", "1-0", "0-1", "1/2-1/2", "*"])
        self.here = QCheckBox("Reaching the board's position")
        self.here.setToolTip("Games reaching the position on the board within their first indexed moves")
        for signal in (self.text.textChanged, self.result.currentIndexChanged, self.here.toggled):
            signal.connect(lambda *_: self.search_changed.emit())
        self.model = GamesModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._double_clicked)
        buttons = QHBoxLayout()
        for b in (self.open, self.add, self.close):
            buttons.addWidget(b)
        buttons.addStretch()
        filters = QHBoxLayout()
        filters.addWidget(self.text, 1)
        filters.addWidget(self.result)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addLayout(filters)
        layout.addWidget(self.here)
        layout.addWidget(self.table, 1)

    def filters(self) -> tuple[str, str, bool]:
        """(text, result or "", reaching the board's position)."""
        result = self.result.currentText()
        return self.text.text(), "" if result == "Any result" else result, self.here.isChecked()

    def show_games(self, db: GameDatabase, games: list[DbGame]) -> None:
        header = self.table.horizontalHeader()
        self.model.set_games(games)
        self.model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
        if not db.games:
            self.status.setText("No database. Open a PGN file of games.")
        else:
            index = (f"; positions indexed for the first {db.index.plies // 2} moves" if db.index is not None else
                     "; indexing positions…" if db.building else "")
            self.status.setText(f"{db.name}: {len(games)} of {len(db.games)} games{index}")
        self.add.setEnabled(bool(db.games))
        self.close.setEnabled(bool(db.games))
        self.here.setEnabled(db.index is not None)

    def show_progress(self, done: int, total: int) -> None:
        self.progress.setVisible(done < total)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"Indexing positions: {done} of {total} games")

    def _double_clicked(self, index: QModelIndex) -> None:
        number = self.model.data(index, Qt.ItemDataRole.UserRole)
        if number is not None:
            self.game_requested.emit(int(number))


class ResultBar(QStyledItemDelegate):
    """Draws a row's White / draw / Black shares as a bar (data: (w, d, b))."""

    def paint(self, painter: QPainter, option, index) -> None:
        shares = index.data(Qt.ItemDataRole.UserRole)
        if not shares:
            super().paint(painter, option, index)
            return
        t = current_theme()
        r = QRectF(option.rect).adjusted(3, 4, -3, -4)
        total = sum(shares) or 1
        x = r.left()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for share, color, text_color in ((shares[0], t.eval_white, "#202020"), (shares[1], "#8a8f99", "#f4f4f4"),
                                         (shares[2], t.eval_black, "#f0f0f0")):
            w = r.width() * share / total
            if w <= 0:
                continue
            part = QRectF(x, r.top(), w, r.height())
            painter.fillRect(part, QColor(color))
            if w > 26:
                painter.setPen(QColor(text_color))
                painter.drawText(part, Qt.AlignmentFlag.AlignCenter, f"{100 * share / total:.0f}%")
            x += w
        painter.restore()


class ExplorerPanel(QWidget):
    """The moves played from the board's position in the database's games:
    how often, and how those games ended. Clicking a move plays it."""

    move_selected = Signal(object)  # Move

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("explorerPanel")
        self.title = QLabel("Explorer: open a database (Database tab) to see the moves played here.")
        self.title.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Move", "Games", "White / draw / Black"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setItemDelegateForColumn(2, ResultBar(self.table))
        self.table.cellClicked.connect(self._clicked)
        self.figurines = False
        self.moves: list[Move] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.title)
        layout.addWidget(self.table, 1)

    def show_position(self, db: GameDatabase, pos: Position, ply: int) -> None:
        """The moves played from `pos` (reached after `ply` moves)."""
        if db.index is None:
            self.title.setText("Explorer: open a database (Database tab) to see the moves played here." if not
                               db.games else "Explorer: indexing the database's positions…")
            self.moves = []
            self.table.setRowCount(0)
            return
        stats = db.index.stats(pos)
        games = len(db.index.games_at(pos))
        if stats:
            self.title.setText(f"{db.name}: {games} games reached this position")
        elif ply >= db.index.plies:
            self.title.setText(f"Explorer: only the first {db.index.plies // 2} moves of each game are indexed")
        else:
            self.title.setText(f"{db.name}: no game played on from here" if games else
                               f"{db.name}: no game reached this position")
        self.moves = [m for m, _ in stats]
        self.table.setRowCount(len(stats))
        for row, (move, s) in enumerate(stats):
            san = pos.san(move)
            item = QTableWidgetItem(figurine_san(san) if self.figurines else san)
            self.table.setItem(row, 0, item)
            count = QTableWidgetItem(str(sum(s)))
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, count)
            bar = QTableWidgetItem("")
            bar.setData(Qt.ItemDataRole.UserRole, (s[0], s[1], s[2]))
            bar.setToolTip(f"White won {s[0]}, drawn {s[1]}, Black won {s[2]}" +
                           (f", unfinished {s[3]}" if s[3] else ""))
            self.table.setItem(row, 2, bar)

    def _clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self.moves):
            self.move_selected.emit(self.moves[row])
