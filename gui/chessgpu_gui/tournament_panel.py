"""The tournament dialog (engines, format, time control, openings,
adjudication, output file) and panel (progress, standings with Elo, the
cross-table and the games)."""

from __future__ import annotations

import pathlib

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from .dialogs import TimeControlEditor
from .profiles import EngineProfile
from .theme import current_theme
from .tournament import (KINDS, Adjudication, GameResult, Standing, TournamentSettings, crosstable, load_openings,
                         los, schedule)

# The balanced openings the engine's own matches use, when present.
BUILT_IN_OPENINGS = pathlib.Path(__file__).resolve().parents[2] / "tests" / "search" / "openings.epd"
OPENING_CHOICES = ("Start position", "Balanced openings (built in)", "File…")


def elo_text(standing: Standing) -> str:
    elo, error = standing.elo()
    if elo is None:
        return "–"
    value = round(elo)
    return f"{value:+d} ± {error:.0f}" if value else f"0 ± {error:.0f}"  # (never "-0")


class TournamentDialog(QDialog):
    """Asks for a tournament's settings and its PGN file."""

    def __init__(self, profiles: list[EngineProfile], last: dict | None = None, folder: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("New tournament")
        last = last or {}
        self.name = QLineEdit(str(last.get("name", "Engine tournament")))
        self.engines = QListWidget()
        chosen = last.get("engines", [p.name for p in profiles[:2]])
        for p in profiles:
            item = QListWidgetItem(p.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if p.name in chosen else Qt.CheckState.Unchecked)
            self.engines.addItem(item)
        self.engines.setMaximumHeight(120)
        self.kind = QComboBox()
        self.kind.addItems([k.capitalize() for k in KINDS])
        self.kind.setCurrentIndex(KINDS.index(last.get("kind", "round robin")) if last.get("kind") in KINDS else 0)
        self.gauntlet = QComboBox()
        self.gauntlet.addItems([p.name for p in profiles])
        if last.get("gauntlet") in [p.name for p in profiles]:
            self.gauntlet.setCurrentText(last["gauntlet"])
        self.rounds = QSpinBox()
        self.rounds.setRange(1, 500)
        self.rounds.setValue(int(last.get("rounds", 2)))
        self.tc = TimeControlEditor(last.get("tc_settings", {"minutes": 1, "increment": 0.6}))
        self.openings = QComboBox()
        self.openings.addItems(OPENING_CHOICES)
        if not BUILT_IN_OPENINGS.exists():
            self.openings.model().item(1).setEnabled(False)
        self.openings.setCurrentIndex(int(last.get("openings_choice", 1 if BUILT_IN_OPENINGS.exists() else 0)))
        self.openings_file = QLineEdit(str(last.get("openings_file", "")))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_openings)
        self.plies = QSpinBox()
        self.plies.setRange(0, 60)
        self.plies.setValue(int(last.get("plies", 8)))
        self.plies.setToolTip("Moves taken from each game of a PGN file")
        self.random_order = QCheckBox("Random order")
        self.random_order.setChecked(str(last.get("random_order", "true")) != "false")
        adj = Adjudication()
        self.adjudicate = QCheckBox("Adjudicate (as fishtest)")
        self.adjudicate.setChecked(str(last.get("adjudicate", "true")) != "false")
        self.resign_cp = self._spin(100, 5000, int(last.get("resign_cp", adj.resign_cp)), " cp")
        self.resign_moves = self._spin(1, 50, int(last.get("resign_moves", adj.resign_moves)), " moves")
        self.draw_cp = self._spin(0, 500, int(last.get("draw_cp", adj.draw_cp)), " cp")
        self.draw_moves = self._spin(1, 100, int(last.get("draw_moves", adj.draw_moves)), " moves")
        self.draw_from = self._spin(1, 500, int(last.get("draw_from", adj.draw_from_move)), "")
        self.max_moves = self._spin(0, 1000, int(last.get("max_moves", adj.max_moves)), "")
        self.max_moves.setSpecialValueText("no limit")
        default_pgn = pathlib.Path(folder or pathlib.Path.home()) / "tournament.pgn"
        self.pgn = QLineEdit(str(last.get("pgn", default_pgn)))
        save_as = QPushButton("Browse…")
        save_as.clicked.connect(self._browse_pgn)
        self.summary = QLabel()
        self.error = QLabel()
        self.error.setStyleSheet(f"color: {current_theme().danger};")
        self.error.setWordWrap(True)

        players = QFormLayout()
        players.addRow("Name", self.name)
        players.addRow("Engines", self.engines)
        players.addRow("Format", self.kind)
        players.addRow("Gauntlet engine", self.gauntlet)
        players.addRow("Rounds", self.rounds)
        openings_box = QGroupBox("Openings")
        o = QFormLayout(openings_box)
        file_row = QHBoxLayout()
        file_row.addWidget(self.openings_file, 1)
        file_row.addWidget(browse)
        o.addRow("From", self.openings)
        o.addRow("File", file_row)
        o.addRow("PGN moves", self.plies)
        o.addRow("", self.random_order)
        adjudication = QGroupBox("Adjudication")
        a = QFormLayout(adjudication)
        a.addRow(self.adjudicate)
        a.addRow("Win: both engines agree on", self._row(self.resign_cp, QLabel("for"), self.resign_moves))
        a.addRow("Draw: both within", self._row(self.draw_cp, QLabel("for"), self.draw_moves))
        a.addRow("Draws from move", self.draw_from)
        a.addRow("Draw after move", self.max_moves)
        out_row = QHBoxLayout()
        out_row.addWidget(self.pgn, 1)
        out_row.addWidget(save_as)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(players)
        layout.addWidget(self.tc)
        layout.addWidget(openings_box)
        layout.addWidget(adjudication)
        out = QFormLayout()
        out.addRow("Games to (PGN)", out_row)
        layout.addLayout(out)
        layout.addWidget(self.summary)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        for signal in (self.engines.itemChanged, self.kind.currentIndexChanged, self.rounds.valueChanged,
                       self.gauntlet.currentIndexChanged, self.openings.currentIndexChanged):
            signal.connect(self._update)
        self._update()

    @staticmethod
    def _spin(lo: int, hi: int, value: int, suffix: str) -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    @staticmethod
    def _row(*widgets) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            row.addWidget(x)
        return w

    def checked_engines(self) -> list[str]:
        names = [self.engines.item(i).text() for i in range(self.engines.count())
                 if self.engines.item(i).checkState() == Qt.CheckState.Checked]
        if KINDS[self.kind.currentIndex()] == "gauntlet":
            first = self.gauntlet.currentText()
            names = [first] + [n for n in names if n != first]
        return names

    def openings_path(self) -> str:
        choice = self.openings.currentIndex()
        if choice == 1:
            return str(BUILT_IN_OPENINGS)
        if choice == 2:
            return self.openings_file.text().strip()
        return ""

    def tournament(self) -> TournamentSettings:
        tc = self.tc.time_control()
        return TournamentSettings(
            name=self.name.text().strip() or "Engine tournament", engines=self.checked_engines(),
            kind=KINDS[self.kind.currentIndex()], rounds=self.rounds.value(),
            tc={"base_ms": tc.base_ms, "inc_ms": tc.inc_ms, "movetime_ms": tc.movetime_ms, "depth": tc.depth,
                "nodes": tc.nodes},
            openings=self.openings_path(), opening_plies=self.plies.value(),
            random_order=self.random_order.isChecked(),
            adjudication=Adjudication(self.resign_cp.value(), self.resign_moves.value(), self.draw_cp.value(),
                                      self.draw_moves.value(), self.draw_from.value(), self.max_moves.value(),
                                      self.adjudicate.isChecked()))

    def settings(self) -> dict:
        """What to preselect next time."""
        return {"name": self.name.text(), "engines": self.checked_engines(), "kind": KINDS[self.kind.currentIndex()],
                "gauntlet": self.gauntlet.currentText(), "rounds": self.rounds.value(),
                "tc_settings": self.tc.settings(), "openings_choice": self.openings.currentIndex(),
                "openings_file": self.openings_file.text(), "plies": self.plies.value(),
                "random_order": "true" if self.random_order.isChecked() else "false",
                "adjudicate": "true" if self.adjudicate.isChecked() else "false",
                "resign_cp": self.resign_cp.value(), "resign_moves": self.resign_moves.value(),
                "draw_cp": self.draw_cp.value(), "draw_moves": self.draw_moves.value(),
                "draw_from": self.draw_from.value(), "max_moves": self.max_moves.value(), "pgn": self.pgn.text()}

    def _update(self, *_args) -> None:
        gauntlet = KINDS[self.kind.currentIndex()] == "gauntlet"
        self.gauntlet.setEnabled(gauntlet)
        self.openings_file.setEnabled(self.openings.currentIndex() == 2)
        engines = self.checked_engines()
        games = len(schedule(engines, KINDS[self.kind.currentIndex()], self.rounds.value(), 1)) \
            if len(engines) >= 2 else 0
        self.summary.setText(f"{len(engines)} engines, {games} games" if games else "Choose two engines or more.")

    def _browse_openings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Openings", self.openings_file.text(),
                                              "Openings (*.epd *.pgn);;All files (*)")
        if path:
            self.openings_file.setText(path)
            self.openings.setCurrentIndex(2)

    def _browse_pgn(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Tournament games", self.pgn.text(), "PGN files (*.pgn)")
        if path:
            self.pgn.setText(path)

    def accept(self) -> None:
        s = self.tournament()
        if len(s.engines) < 2:
            self.error.setText("Choose two engines or more.")
            return
        if s.kind == "gauntlet" and len(s.engines) < 2:
            self.error.setText("A gauntlet needs an opponent for its engine.")
            return
        try:
            load_openings(s.openings, s.opening_plies)
        except (OSError, ValueError) as e:
            self.error.setText(f"Openings: {e}")
            return
        if not self.pgn.text().strip():
            self.error.setText("Choose a file for the games.")
            return
        super().accept()


class TournamentPanel(QWidget):
    """A tournament's progress, standings, cross-table and games."""

    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    new_requested = Signal()
    game_requested = Signal(int)  # a finished game's number (to show it)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("tournamentPanel")
        self.status = QLabel("No tournament. Start one from the Tournament menu.")
        self.status.setWordWrap(True)
        self.pause = QPushButton("Pause")
        self.resume = QPushButton("Resume")
        self.stop = QPushButton("Stop")
        self.new = QPushButton("New tournament…")
        for button, signal in ((self.pause, self.pause_requested), (self.resume, self.resume_requested),
                               (self.stop, self.stop_requested), (self.new, self.new_requested)):
            button.clicked.connect(signal.emit)
        buttons = QHBoxLayout()
        for b in (self.pause, self.resume, self.stop):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(self.new)
        self.table = self._table(["Engine", "Games", "+", "=", "−", "Points", "Score", "Elo"])
        self.cross = self._table([])
        self.cross.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # (equal columns)
        self.cross.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # (stretched: never needed)
        self.games = self._table(["#", "White", "Black", "Result", "How", "Moves"])
        self.games.cellDoubleClicked.connect(self._game_clicked)
        self.views = QTabWidget()
        self.views.addTab(self.table, "Standings")
        self.views.addTab(self.cross, "Cross-table")
        self.views.addTab(self.games, "Games")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        layout.addWidget(self.views, 1)
        self.set_state("idle")

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.verticalHeader().setDefaultSectionSize(22)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.setAlternatingRowColors(True)
        return t

    def set_state(self, state: str, text: str = "") -> None:
        """"idle", "running" or "paused"."""
        self.pause.setEnabled(state == "running")
        self.resume.setEnabled(state == "paused")
        self.stop.setEnabled(state != "idle")
        self.new.setEnabled(state == "idle")
        if text:
            self.status.setText(text)

    def show_results(self, engines: list[str], table: list[Standing], results: list[GameResult]) -> None:
        self.table.setRowCount(len(table))
        for row, s in enumerate(table):
            values = [s.engine, str(s.games), str(s.wins), str(s.draws), str(s.losses), f"{s.points:g}",
                      f"{100 * s.score:.1f}%" if s.games else "–", elo_text(s)]
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(f"LOS {100 * los(s.wins, s.losses):.0f}%" if col == 7 else "")
                self.table.setItem(row, col, item)
        cross = crosstable(engines, results)
        self.cross.setColumnCount(len(engines))
        self.cross.setRowCount(len(engines))
        self.cross.setHorizontalHeaderLabels([f"{i + 1}" for i in range(len(engines))])
        self.cross.setVerticalHeaderLabels([f"{i + 1}. {e}" for i, e in enumerate(engines)])
        for r, a in enumerate(engines):
            for c, b in enumerate(engines):
                if a == b:
                    text = "·"
                else:
                    points, games = cross.get((a, b), (0.0, 0))
                    text = f"{points:g}/{games}" if games else ""
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.cross.setItem(r, c, item)
        self.games.setRowCount(len(results))
        for row, g in enumerate(sorted(results, key=lambda g: g.number)):
            for col, v in enumerate((str(g.number), g.white, g.black, g.result, g.reason, str((g.plies + 1) // 2))):
                item = QTableWidgetItem(v)
                item.setData(Qt.ItemDataRole.UserRole, g.number)
                self.games.setItem(row, col, item)

    def _game_clicked(self, row: int, _col: int) -> None:
        item = self.games.item(row, 0)
        if item is not None:
            self.game_requested.emit(int(item.data(Qt.ItemDataRole.UserRole)))
