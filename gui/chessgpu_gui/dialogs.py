"""The new-game dialog, the board editor and the settings dialog."""

from __future__ import annotations

from dataclasses import dataclass, fields

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QRadioButton, QSlider, QSpinBox, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .board_view import BoardView
from .game import TimeControl
from .pieces import DEFAULT_PIECE_SET, PIECE_SETS, PieceSet
from .profiles import EngineProfile
from .rules import BLACK, EMPTY, START_FEN, WHITE, FenError, Position, file_of, rank_of, square, square_name
from .theme import BOARD_THEMES, DEFAULT_BOARD_THEME, DEFAULT_THEME, THEMES, current_theme

ANIMATION_SPEEDS = {"Off": 0, "Fast": 90, "Normal": 160, "Slow": 260}


@dataclass
class ViewSettings:
    """How the GUI looks and behaves; stored in the settings file under
    these field names (piece_set and coordinates as before)."""

    theme: str = DEFAULT_THEME
    board_theme: str = DEFAULT_BOARD_THEME
    piece_set: str = DEFAULT_PIECE_SET
    animation_ms: int = ANIMATION_SPEEDS["Normal"]
    sounds: bool = True
    volume: int = 70  # 0..100
    show_legal: bool = True
    coordinates: bool = True
    highlight_last: bool = True
    premoves: bool = True
    auto_queen: bool = False
    figurines: bool = False

    @staticmethod
    def load(settings: QSettings) -> ViewSettings:
        """The stored settings; missing or unreadable values keep their default."""
        out = ViewSettings()
        for f in fields(ViewSettings):
            raw = settings.value(f.name)
            if raw is None:
                continue
            default = getattr(out, f.name)
            try:
                if isinstance(default, bool):
                    value = raw if isinstance(raw, bool) else str(raw).lower() == "true"
                elif isinstance(default, int):
                    value = int(raw)
                else:
                    value = str(raw)
            except (TypeError, ValueError):
                continue
            setattr(out, f.name, value)
        if out.theme not in THEMES:
            out.theme = DEFAULT_THEME
        if out.board_theme not in BOARD_THEMES:
            out.board_theme = DEFAULT_BOARD_THEME
        if out.piece_set not in PIECE_SETS:
            out.piece_set = DEFAULT_PIECE_SET
        out.volume = max(0, min(100, out.volume))
        out.animation_ms = max(0, min(1000, out.animation_ms))
        return out

    def save(self, settings: QSettings) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            settings.setValue(f.name, ("true" if value else "false") if isinstance(value, bool) else value)


def board_theme_icon(name: str, size: int = 32) -> QIcon:
    """Four squares of a board theme, for choosing one."""
    t = BOARD_THEMES[name]
    pm = QPixmap(size, size)
    p = QPainter(pm)
    half = size // 2
    for i, (x, y) in enumerate(((0, 0), (half, 0), (0, half), (half, half))):
        p.fillRect(x, y, half, half, QColor(t.light if i in (0, 3) else t.dark))
    p.end()
    return QIcon(pm)


class SettingsDialog(QDialog):
    """Theme, board, pieces, animation, sounds and play options."""

    def __init__(self, current: ViewSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.theme = QComboBox()
        for name, t in THEMES.items():
            self.theme.addItem(t.title, name)
        self.board_theme = QComboBox()
        self.board_theme.setIconSize(QSize(28, 28))
        for name, t in BOARD_THEMES.items():
            self.board_theme.addItem(board_theme_icon(name), t.title, name)
        self.piece_set = QComboBox()
        for name, cls in PIECE_SETS.items():
            self.piece_set.addItem(cls.title, name)
        self.animation = QComboBox()
        for label, ms in ANIMATION_SPEEDS.items():
            self.animation.addItem(label, ms)
        self.sounds = QCheckBox("Play sounds")
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.show_legal = QCheckBox("Show legal moves of the selected piece")
        self.coordinates = QCheckBox("Show coordinates")
        self.highlight_last = QCheckBox("Highlight the last move")
        self.premoves = QCheckBox("Allow premoves while the engine thinks")
        self.auto_queen = QCheckBox("Promote to a queen without asking")
        self.figurines = QCheckBox("Figurine notation (♘f3)")

        look = QGroupBox("Appearance")
        look_form = QFormLayout(look)
        look_form.addRow("Theme", self.theme)
        look_form.addRow("Board", self.board_theme)
        look_form.addRow("Pieces", self.piece_set)
        look_form.addRow("Animation", self.animation)
        look_form.addRow(self.coordinates)
        look_form.addRow(self.highlight_last)
        look_form.addRow(self.figurines)
        sound = QGroupBox("Sound")
        sound_form = QFormLayout(sound)
        sound_form.addRow(self.sounds)
        sound_form.addRow("Volume", self.volume)
        play = QGroupBox("Play")
        play_layout = QVBoxLayout(play)
        for box in (self.show_legal, self.premoves, self.auto_queen):
            play_layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        for w in (look, sound, play):
            layout.addWidget(w)
        layout.addWidget(buttons)
        self.sounds.toggled.connect(self.volume.setEnabled)
        self.set_settings(current)

    def set_settings(self, s: ViewSettings) -> None:
        for combo, value in ((self.theme, s.theme), (self.board_theme, s.board_theme), (self.piece_set, s.piece_set)):
            combo.setCurrentIndex(max(0, combo.findData(value)))
        index = self.animation.findData(s.animation_ms)
        if index < 0:  # a custom speed from the settings file
            self.animation.addItem(f"{s.animation_ms} ms", s.animation_ms)
            index = self.animation.count() - 1
        self.animation.setCurrentIndex(index)
        self.sounds.setChecked(s.sounds)
        self.volume.setValue(s.volume)
        self.volume.setEnabled(s.sounds)
        for box, value in ((self.show_legal, s.show_legal), (self.coordinates, s.coordinates),
                           (self.highlight_last, s.highlight_last), (self.premoves, s.premoves),
                           (self.auto_queen, s.auto_queen), (self.figurines, s.figurines)):
            box.setChecked(value)

    def result_settings(self) -> ViewSettings:
        return ViewSettings(
            theme=self.theme.currentData(), board_theme=self.board_theme.currentData(),
            piece_set=self.piece_set.currentData(), animation_ms=int(self.animation.currentData()),
            sounds=self.sounds.isChecked(), volume=self.volume.value(), show_legal=self.show_legal.isChecked(),
            coordinates=self.coordinates.isChecked(), highlight_last=self.highlight_last.isChecked(),
            premoves=self.premoves.isChecked(), auto_queen=self.auto_queen.isChecked(),
            figurines=self.figurines.isChecked())


@dataclass
class NewGameChoice:
    white: EngineProfile | None
    black: EngineProfile | None
    tc: TimeControl
    start: Position
    book_plies: int = 0  # engines play from the opening book for this many moves


class TimeControlEditor(QGroupBox):
    """A time control: a clock (minutes + increment), a time per move, a
    depth or a node count."""

    MODES = ("Clock", "Time per move", "Depth", "Nodes")

    def __init__(self, last: dict | None = None, note: str = "", parent: QWidget | None = None):
        super().__init__("Time control", parent)
        last = last or {}
        self.mode = QComboBox()
        self.mode.addItems(self.MODES)
        self.minutes = QDoubleSpinBox()
        self.minutes.setRange(0.05, 600)
        self.minutes.setDecimals(2)
        self.minutes.setSuffix(" min")
        self.minutes.setValue(float(last.get("minutes", 5)))
        self.increment = QDoubleSpinBox()
        self.increment.setRange(0, 600)
        self.increment.setDecimals(2)
        self.increment.setSuffix(" s")
        self.increment.setValue(float(last.get("increment", 3)))
        self.movetime = QDoubleSpinBox()
        self.movetime.setRange(0.01, 3600)
        self.movetime.setSuffix(" s")
        self.movetime.setValue(float(last.get("movetime", 1)))
        self.depth = QSpinBox()
        self.depth.setRange(1, 63)
        self.depth.setValue(int(last.get("depth", 8)))
        self.nodes = QSpinBox()
        self.nodes.setRange(1, 2**31 - 1)
        self.nodes.setValue(int(last.get("nodes", 1_000_000)))
        pages = QStackedWidget()
        clock = QWidget()
        clock_row = QHBoxLayout(clock)
        clock_row.setContentsMargins(0, 0, 0, 0)
        clock_row.addWidget(self.minutes)
        clock_row.addWidget(QLabel("+"))
        clock_row.addWidget(self.increment)
        for page in (clock, self.movetime, self.depth, self.nodes):
            pages.addWidget(page)
        self.mode.currentIndexChanged.connect(pages.setCurrentIndex)
        self.mode.setCurrentIndex(max(0, self.MODES.index(last.get("mode", "Clock"))
                                      if last.get("mode") in self.MODES else 0))
        layout = QFormLayout(self)
        layout.addRow("Limit", self.mode)
        layout.addRow("", pages)
        self.note = QLabel(note)
        self.note.setWordWrap(True)
        if note:
            layout.addRow(self.note)

    def time_control(self) -> TimeControl:
        mode = self.mode.currentText()
        if mode == "Clock":
            return TimeControl(base_ms=round(self.minutes.value() * 60_000), inc_ms=round(self.increment.value() * 1000))
        if mode == "Time per move":
            return TimeControl(movetime_ms=round(self.movetime.value() * 1000))
        if mode == "Depth":
            return TimeControl(depth=self.depth.value())
        return TimeControl(nodes=self.nodes.value())

    def settings(self) -> dict:
        return {"mode": self.mode.currentText(), "minutes": self.minutes.value(), "increment": self.increment.value(),
                "movetime": self.movetime.value(), "depth": self.depth.value(), "nodes": self.nodes.value()}


class NewGameDialog(QDialog):
    """Players (human or an engine profile), time control, start position."""

    MODES = TimeControlEditor.MODES

    def __init__(self, profiles: list[EngineProfile], current: Position, last: dict | None = None,
                 parent: QWidget | None = None, book: str = ""):
        super().__init__(parent)
        self.setWindowTitle("New game")
        self.profiles = profiles
        self.current = current
        last = last or {}
        self.book = QCheckBox(f"Engines play from the book ({book}) for the first" if book else
                              "Engines play from an opening book (open a database first)")
        self.book.setEnabled(bool(book))
        self.book.setChecked(bool(book) and str(last.get("book", "false")) == "true")
        self.book_moves = QSpinBox()
        self.book_moves.setRange(1, 40)
        self.book_moves.setSuffix(" moves")
        self.book_moves.setValue(int(last.get("book_moves", 8)))
        self.book_moves.setEnabled(bool(book))
        self.players = []
        players = QFormLayout()
        for color, label in ((WHITE, "White"), (BLACK, "Black")):
            combo = QComboBox()
            combo.addItem("Human")
            combo.addItems([p.name for p in profiles])
            wanted = last.get("white" if color == WHITE else "black", "Human" if color == WHITE else
                              (profiles[0].name if profiles else "Human"))
            index = combo.findText(wanted)
            combo.setCurrentIndex(index if index >= 0 else 0)
            self.players.append(combo)
            players.addRow(label, combo)

        tc_box = TimeControlEditor(last, "Depth and node limits apply to engines; humans play untimed.")
        self.tc_editor = tc_box
        self.mode, self.minutes, self.increment = tc_box.mode, tc_box.minutes, tc_box.increment
        self.movetime, self.depth, self.nodes, self.tc_note = tc_box.movetime, tc_box.depth, tc_box.nodes, tc_box.note

        self.start_initial = QRadioButton("Initial position")
        self.start_current = QRadioButton("Current board position")
        self.start_fen = QRadioButton("FEN")
        self.fen = QLineEdit(START_FEN)
        self.start_initial.setChecked(True)
        start_box = QGroupBox("Start position")
        start_layout = QVBoxLayout(start_box)
        for w in (self.start_initial, self.start_current, self.start_fen, self.fen):
            start_layout.addWidget(w)
        self.error = QLabel("")
        self.error.setStyleSheet(f"color: {current_theme().danger};")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(players)
        layout.addWidget(tc_box)
        book_row = QHBoxLayout()
        book_row.addWidget(self.book)
        book_row.addWidget(self.book_moves)
        book_row.addStretch()
        layout.addLayout(book_row)
        layout.addWidget(start_box)
        layout.addWidget(self.error)
        layout.addWidget(buttons)

    def player(self, color: int) -> EngineProfile | None:
        i = self.players[color].currentIndex()
        return None if i == 0 else self.profiles[i - 1]

    def time_control(self) -> TimeControl:
        return self.tc_editor.time_control()

    def start_position(self) -> Position:
        if self.start_current.isChecked():
            return self.current
        if self.start_fen.isChecked():
            return Position.from_fen(self.fen.text().strip())
        return Position.start()

    def choice(self) -> NewGameChoice:
        return NewGameChoice(self.player(WHITE), self.player(BLACK), self.time_control(), self.start_position(),
                             2 * self.book_moves.value() if self.book.isChecked() else 0)

    def settings(self) -> dict:
        """What to preselect next time."""
        return {"white": self.players[WHITE].currentText(), "black": self.players[BLACK].currentText(),
                **self.tc_editor.settings(), "book": "true" if self.book.isChecked() else "false",
                "book_moves": self.book_moves.value()}

    def accept(self) -> None:
        try:
            self.start_position()
        except FenError as e:
            self.error.setText(f"Invalid FEN: {e}")
            return
        if not self.start_position().legal_moves():
            self.error.setText("The start position is already over (no legal moves).")
            return
        super().accept()


_CASTLING_NEEDS = {"K": (4, 7, "K", "R"), "Q": (4, 0, "K", "R"), "k": (60, 63, "k", "r"), "q": (60, 56, "k", "r")}


class BoardEditor(QDialog):
    """Set up a position: place pieces, choose the side to move, castling
    rights and en passant square, or type a FEN. OK needs a valid position
    (the same rules the engine applies)."""

    def __init__(self, start: Position, piece_set: PieceSet, flipped: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit position")
        self.cells = list(start.board)
        self.board = BoardView(piece_set)
        self.board.edit_mode = True
        self.board.set_flipped(flipped)
        self.board.setMinimumSize(400, 400)
        self.board.square_clicked.connect(self._clicked)
        self.brush = "P"
        palette = QGridLayout()
        self.brushes = QButtonGroup(self)
        for i, piece in enumerate("KQRBNP" "kqrbnp"):
            b = QToolButton()
            b.setCheckable(True)
            b.setIcon(QIcon(piece_set.pixmap(piece, 40)))
            b.setIconSize(QSize(36, 36))
            b.setToolTip(piece)
            b.clicked.connect(lambda _c=False, p=piece: self._set_brush(p))
            self.brushes.addButton(b)
            palette.addWidget(b, i // 6, i % 6)
            if piece == "P":
                b.setChecked(True)
        eraser = QToolButton()
        eraser.setText("Erase")
        eraser.setCheckable(True)
        eraser.clicked.connect(lambda: self._set_brush(EMPTY))
        self.brushes.addButton(eraser)
        palette.addWidget(eraser, 2, 0, 1, 2)
        self.side = QComboBox()
        self.side.addItems(["White to move", "Black to move"])
        self.side.setCurrentIndex(start.turn)
        self.castle = {c: QCheckBox(c) for c in "KQkq"}
        for c, box in self.castle.items():
            box.setChecked(c in start.castling)
        self.ep = QComboBox()
        self.halfmove = QSpinBox()
        self.halfmove.setRange(0, 9999)
        self.halfmove.setValue(start.halfmove)
        self.fullmove = QSpinBox()
        self.fullmove.setRange(1, 9999)
        self.fullmove.setValue(max(1, start.fullmove))
        self.fen = QLineEdit()
        self.fen.editingFinished.connect(self._fen_edited)
        self.problem = QLabel("")
        self.problem.setStyleSheet(f"color: {current_theme().danger};")
        self.problem.setWordWrap(True)
        tools = QHBoxLayout()
        for text, slot in (("Clear", self.clear), ("Initial position", self.initial),
                           ("Flip board", lambda: self.board.set_flipped(not self.board.flipped))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            tools.addWidget(b)
        castle_row = QHBoxLayout()
        for box in self.castle.values():
            castle_row.addWidget(box)
        form = QFormLayout()
        form.addRow("Side to move", self.side)
        form.addRow("Castling", castle_row)
        form.addRow("En passant", self.ep)
        form.addRow("Halfmove clock", self.halfmove)
        form.addRow("Move number", self.fullmove)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        side = QVBoxLayout()
        side.addLayout(palette)
        side.addLayout(form)
        side.addLayout(tools)
        side.addStretch()
        row = QHBoxLayout()
        row.addWidget(self.board, 1)
        row.addLayout(side)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(QLabel("FEN"))
        layout.addWidget(self.fen)
        layout.addWidget(self.problem)
        layout.addWidget(self.buttons)
        for w in (self.side, self.ep):
            w.currentIndexChanged.connect(self._changed)
        for box in self.castle.values():
            box.toggled.connect(self._changed)
        for spin in (self.halfmove, self.fullmove):
            spin.valueChanged.connect(self._changed)
        self._fill_ep(start.ep)
        self._changed()

    def _set_brush(self, piece: str) -> None:
        self.brush = piece

    def _clicked(self, sq: int, button) -> None:
        if button == Qt.MouseButton.RightButton or self.brush == EMPTY or self.cells[sq] == self.brush:
            self.cells[sq] = EMPTY
        else:
            self.cells[sq] = self.brush
        self._changed()

    def clear(self) -> None:
        self.cells = [EMPTY] * 64
        self._changed()

    def initial(self) -> None:
        self.set_position(Position.start())

    def set_position(self, pos: Position) -> None:
        self.cells = list(pos.board)
        self.side.setCurrentIndex(pos.turn)
        for c, box in self.castle.items():
            box.setChecked(c in pos.castling)
        self.halfmove.setValue(pos.halfmove)
        self.fullmove.setValue(max(1, pos.fullmove))
        self._fill_ep(pos.ep)
        self._changed()

    def _fill_ep(self, keep: int | None) -> None:
        """En passant choices: squares behind a pawn that could just have
        made a double push."""
        turn = self.side.currentIndex()
        rank = 5 if turn == WHITE else 2
        pawn = "p" if turn == WHITE else "P"
        step = -8 if turn == WHITE else 8
        choices = [s for s in (square(f, rank) for f in range(8))
                   if self.cells[s] == EMPTY and self.cells[s - step] == EMPTY and self.cells[s + step] == pawn]
        self.ep.blockSignals(True)
        self.ep.clear()
        self.ep.addItem("–", None)
        for s in choices:
            self.ep.addItem(square_name(s), s)
        index = self.ep.findData(keep) if keep in choices else 0
        self.ep.setCurrentIndex(max(0, index))
        self.ep.blockSignals(False)

    def draft(self) -> Position:
        """The position as set up (possibly invalid)."""
        castling = "".join(c for c, box in self.castle.items() if box.isChecked())
        return Position("".join(self.cells), self.side.currentIndex(), castling, self.ep.currentData(),
                        self.halfmove.value(), self.fullmove.value())

    def _sync_castling(self) -> None:
        """Castling rights need the king and the rook at home: impossible
        ones are unchecked and disabled."""
        for right, (king_sq, rook_sq, king, rook) in _CASTLING_NEEDS.items():
            possible = self.cells[king_sq] == king and self.cells[rook_sq] == rook
            box = self.castle[right]
            box.blockSignals(True)
            box.setEnabled(possible)
            if not possible:
                box.setChecked(False)
            box.blockSignals(False)

    def _changed(self, *_args) -> None:
        self._sync_castling()
        self._fill_ep(self.ep.currentData())
        pos = self.draft()
        self.board.set_position(pos)
        self.fen.setText(pos.fen())
        problem = pos.problem()
        self.problem.setText(problem)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not problem)

    def _fen_edited(self) -> None:
        try:
            pos = Position.from_fen(self.fen.text().strip(), validate=False)
        except FenError as e:
            self.problem.setText(f"Invalid FEN: {e}")
            return
        self.set_position(pos)

    def result_position(self) -> Position:
        return Position.from_fen(self.draft().fen())
