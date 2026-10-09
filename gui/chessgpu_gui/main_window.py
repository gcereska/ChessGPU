"""The main window: board between the two player cards, the move tree
(variations and comments) with navigation, engine output, eval bar and
graph; a toolbar and menus for games, PGN and FEN, analysis, engines, the
view and settings.

The window shows one node of the game tree (view_node). Navigation moves
along the line through it: back to the start, forward along its main
continuation; Up and Down switch between the moves played from the same
position. Moves made on the board outside a game add variations.

Look and feel come from ViewSettings (dialogs.py): theme (theme.py), board
colors, piece set, animation speed, sounds (sounds.py), and play options
(legal-move dots, premoves, automatic queening, figurine notation).
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import QSettings, QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from .board_view import BoardView
from .dialogs import BoardEditor, NewGameDialog, SettingsDialog, ViewSettings
from .game import COLOR_NAMES, AnalysisController, GameController, GameRecord, Node
from .icons import icon
from .move_tree import CommentBox, MoveTreeView, line_text, toggle_nag
from .options_dialog import ProfilesDialog
from .panels import BoardArea, EngineOutput, EvalBar, EvalGraph, NavigationBar, PlayerCard, node_score, white_score
from .pgn import PgnError, read_pgn, write_pgn
from .pieces import PIECE_SETS, make_piece_set
from .profiles import EngineProfile, default_profiles, profiles_from_json, profiles_to_json
from .review import ReviewController, annotate, judge, summarize
from .review_panel import ReviewDialog, ReviewPanel, kind_color
from .tournament import TournamentController
from .tournament_panel import TournamentDialog, TournamentPanel
from .database import GameDatabase
from .database_panel import DatabasePanel, ExplorerPanel
from .rules import BLACK, WHITE, FenError, IllegalMoveError, Move, Position
from .sounds import SoundPlayer, move_sound
from .theme import BOARD_THEMES, ENGINE_ARROW, THEMES, THREAT_ARROW, apply_theme, current_theme

ABOUT = """<b>ChessGPU GUI</b><br>
A front end for UCI engines, made for the ChessGPU engine (C++/CUDA, alpha-beta search on the GPU).<br><br>
Piece set "cburnett" by Cburnett, from Wikimedia Commons, used under the BSD 3-clause license
(see chessgpu_gui/assets/pieces/cburnett/LICENSE.txt). The "vector" pieces, the sounds and the chess rules
are this project's own."""

LOW_TIME_WARNING_MS = 10_000  # a human's clock ticks (sound) once when it falls below this
MAX_ANALYSIS_LINES = 5
HINT_MS = 1500  # how long the analysis engine thinks about a hint
LINE_ALPHAS = (170, 110, 80, 60, 45)  # arrows of the best lines, from the best


def line_arrow_colors(base: tuple, count: int) -> list[tuple]:
    """Arrow colors for `count` lines: the best opaque, the others fainter."""
    return [base[:3] + (LINE_ALPHAS[min(i, len(LINE_ALPHAS) - 1)],) for i in range(count)]


def default_settings() -> QSettings:
    folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    return QSettings(str(pathlib.Path(folder) / "gui.ini"), QSettings.Format.IniFormat)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None, sounds: SoundPlayer | None = None):
        super().__init__()
        self.settings = settings or default_settings()
        self.view = ViewSettings.load(self.settings)
        self.profiles: list[EngineProfile] = profiles_from_json(self.settings.value("profiles", "")) or \
            default_profiles()
        self.piece_set_name = self.view.piece_set
        self.analysis_profile_name = str(self.settings.value("analysis_engine", self.profiles[0].name
                                                             if self.profiles else ""))
        self.game = GameController(self)
        self.analysis = AnalysisController(self)
        self.review = ReviewController(self)
        self.tournament = TournamentController(self.game, self._profile_named, self)
        self.database = GameDatabase(self)
        self.judgements = []  # the review's verdicts on the main line (from node.review)
        self._review_signature: list = []
        self._review_record: GameRecord | None = None  # the game a running review is for
        self._review_annotate = True
        self.view_node: Node = self.game.record.root  # the position shown
        self._follow = True  # show new main-line moves as they come (unless browsing elsewhere)
        self._analysed: tuple | None = None  # (start fen, moves) under analysis
        self._analysis_root: Position | None = None  # the position analysed (after a pass in threat mode)
        self.threat_mode = False  # analyse what the opponent threatens (the position after a pass)
        self._hint_for: Node | None = None  # the game position a hint was asked for
        try:
            lines = int(self.settings.value("analysis_lines", 1))
        except (TypeError, ValueError):
            lines = 1
        self.analysis_lines = max(1, min(MAX_ANALYSIS_LINES, lines))
        self._seen_moves = 0  # moves of the record when the window last looked (for sounds)
        self._seen_status = self.game.status
        self._warned: set[int] = set()  # colors whose low-time tick has sounded this game
        self.sounds = sounds or self._default_sounds()

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.view.theme)
        piece_set = make_piece_set(self.piece_set_name)
        self.board = BoardView(piece_set)
        self.board.set_flipped(self.settings.value("flipped", "false") == "true")
        self.eval_bar = EvalBar(self.board)
        self.eval_bar.set_flipped(self.board.flipped)
        self.board.align_left = True
        self.board.resized.connect(self.eval_bar.update)
        self.cards = {WHITE: PlayerCard(self.game, piece_set, WHITE), BLACK: PlayerCard(self.game, piece_set, BLACK)}
        self.move_tree = MoveTreeView()
        self.comment_box = CommentBox()
        self._comment_timer = QTimer(self)  # redraws the tree a moment after comment edits
        self._comment_timer.setSingleShot(True)
        self._comment_timer.setInterval(300)
        self._comment_timer.timeout.connect(self.move_tree.refresh)
        self.navigation = NavigationBar()
        self.outputs = {"white": EngineOutput("White engine"), "black": EngineOutput("Black engine"),
                        "analysis": EngineOutput("Analysis")}
        self.lines_box = QSpinBox()
        self.lines_box.setRange(1, MAX_ANALYSIS_LINES)
        self.lines_box.setValue(self.analysis_lines)
        self.lines_box.setPrefix("Lines ")
        self.lines_box.setToolTip("How many of the best moves the analysis shows (engines with MultiPV)")
        self.outputs["analysis"].header.addWidget(self.lines_box)
        self.outputs["analysis"].set_max_lines(self.analysis_lines)
        self.analysis.lines = self.analysis_lines
        self.graph = EvalGraph()
        self.review_panel = ReviewPanel()
        self.tournament_panel = TournamentPanel()
        self.explorer_panel = ExplorerPanel()
        self.database_panel = DatabasePanel()
        self.status = QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self._apply_view(self.view)
        self._build_layout()
        self._build_log_dock()
        self._build_menus()
        self._build_toolbar()
        self._card_timer = QTimer(self)
        self._card_timer.timeout.connect(self._tick)
        self._card_timer.start(100)

        self.board.move_made.connect(self._board_move)
        self.board.shapes_changed.connect(self._shapes_drawn)
        self.move_tree.node_selected.connect(self.go_to_node)
        self.move_tree.action_requested.connect(self._tree_action)
        self.comment_box.edited.connect(self._comment_timer.start)
        self.graph.ply_selected.connect(self.go_to)
        self.navigation.first.connect(lambda: self.go_to(0))
        self.navigation.previous.connect(lambda: self.go_to(self.view_ply - 1))
        self.navigation.next.connect(lambda: self.go_to(self.view_ply + 1))
        self.navigation.last.connect(self.go_to_end)
        self.game.changed.connect(self._game_changed)
        self.game.engine_info.connect(self._engine_info)
        self.game.game_over.connect(self._game_over)
        self.game.message.connect(self.show_message)
        self.game.status_changed.connect(lambda _s: self._update_panels())
        self.game.engine_log.connect(self._log)
        self.analysis.info.connect(self._analysis_info)
        self.analysis.message.connect(self.show_message)
        self.analysis.running_changed.connect(self._analysis_running)
        self.analysis.engine_log.connect(self._log)
        self.analysis.hint_ready.connect(self._hint_ready)
        self.review.progress.connect(self._review_progress)
        self.review.finished.connect(self._review_finished)
        self.review.message.connect(self.show_message)
        self.review.engine_log.connect(self._log)
        self.review_panel.node_selected.connect(self.go_to_node)
        self.review_panel.cancel_requested.connect(self.review.cancel)
        self.review_panel.review_requested.connect(self.start_review)
        self.tournament.changed.connect(self._tournament_changed)
        self.tournament.game_started.connect(self._tournament_game_started)
        self.tournament.finished.connect(self._tournament_finished)
        self.tournament.message.connect(self.show_message)
        self.tournament_panel.pause_requested.connect(self.tournament.pause)
        self.tournament_panel.resume_requested.connect(self.tournament.resume)
        self.tournament_panel.stop_requested.connect(self.stop_tournament)
        self.tournament_panel.new_requested.connect(self.new_tournament)
        self.tournament_panel.game_requested.connect(self.show_tournament_game)
        self.database.changed.connect(self._database_changed)
        self.database.index_progress.connect(self.database_panel.show_progress)
        self.database.index_ready.connect(self._database_changed)
        self.database.index_failed.connect(lambda why: (self.show_message(f"Couldn't index the database: {why}"),
                                                        self._database_changed()))
        self.database.index_message.connect(self.show_message)
        self.database.index_ready.connect(lambda: self.show_message(
            f"{self.database.name}: positions of {self.database.index.indexed} games indexed"
            if self.database.index is not None else ""))
        self.database_panel.open_requested.connect(self.open_database)
        self.database_panel.add_requested.connect(self.add_database_file)
        self.database_panel.close_requested.connect(self.database.clear)
        self.database_panel.game_requested.connect(self.open_database_game)
        self.database_panel.search_changed.connect(self._search_database)
        self.explorer_panel.move_selected.connect(self._explorer_move)
        self.outputs["analysis"].pv_clicked.connect(self._pv_clicked)
        self.lines_box.valueChanged.connect(self.set_analysis_lines)

        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1280, 860)
        self.setWindowTitle("ChessGPU")
        self._refresh()

    def _default_sounds(self) -> SoundPlayer:
        """Sound files next to the settings file; silent offscreen (tests)."""
        folder = pathlib.Path(self.settings.fileName()).parent / "sounds"
        player = SoundPlayer(folder, enabled=self.view.sounds, volume=self.view.volume / 100)
        if QGuiApplication.platformName() == "offscreen":
            player.backend = None
        return player

    # --- Layout -------------------------------------------------------------------

    def _build_layout(self) -> None:
        top_color = WHITE if self.board.flipped else BLACK
        self.board_area = BoardArea(self.board, self.eval_bar, self.cards[top_color], self.cards[1 - top_color])
        board_area = self.board_area

        moves_page = QWidget()
        moves_column = QVBoxLayout(moves_page)
        moves_column.setContentsMargins(6, 6, 6, 6)
        moves_column.addWidget(self.move_tree, 1)
        moves_column.addWidget(self.comment_box)
        moves_column.addWidget(self.navigation)
        for out in self.outputs.values():
            moves_column.addWidget(out)
        self.tabs = QTabWidget()
        self.tabs.addTab(moves_page, "Moves")
        self.tabs.addTab(self.review_panel, "Review")
        self.tabs.addTab(self.tournament_panel, "Tournament")
        self.tabs.addTab(self.explorer_panel, "Explorer")
        self.tabs.addTab(self.database_panel, "Database")
        side = QWidget()
        side.setMinimumWidth(340)
        side_column = QVBoxLayout(side)
        side_column.setContentsMargins(4, 8, 8, 8)
        side_column.addWidget(self.tabs)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(board_area)
        top.addWidget(side)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.graph)
        main.setStretchFactor(0, 6)
        main.setStretchFactor(1, 1)
        self.setCentralWidget(main)

    def _place_cards(self) -> None:
        """The player at the top of the board above it, the other below."""
        top_color = WHITE if self.board.flipped else BLACK
        self.board_area.set_cards(self.cards[top_color], self.cards[1 - top_color])

    def _build_log_dock(self) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_dock = QDockWidget("Engine log", self)
        self.log_dock.setObjectName("engine_log")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _action(self, menu, text, slot, shortcut=None, checkable=False) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        game = bar.addMenu("&Game")
        self.new_action = self._action(game, "&New game…", self.new_game, "Ctrl+N")
        self.open_action = self._action(game, "&Open PGN…", self.open_pgn, "Ctrl+O")
        self.save_action = self._action(game, "&Save PGN…", self.save_pgn, "Ctrl+S")
        game.addSeparator()
        self._action(game, "&Copy FEN", self.copy_fen, "Ctrl+Shift+C")
        self._action(game, "&Paste FEN", self.paste_fen, "Ctrl+Shift+V")
        self._action(game, "&Edit position…", self.edit_position, "Ctrl+E")
        game.addSeparator()
        self.takeback_action = self._action(game, "&Take back", self.takeback, "Ctrl+Z")
        self.move_now_action = self._action(game, "&Move now", self.game.move_now, "Ctrl+M")
        self.resign_action = self._action(game, "&Resign", self.resign)
        self.abort_action = self._action(game, "&Abort", self.game.abort)
        game.addSeparator()
        self._action(game, "&Quit", self.close, "Ctrl+Q")

        nav = bar.addMenu("&Navigate")
        self._action(nav, "Start", lambda: self.go_to(0), "Home")
        self._action(nav, "Back", lambda: self.go_to(self.view_ply - 1), "Left")
        self._action(nav, "Forward", lambda: self.go_to(self.view_ply + 1), "Right")
        self._action(nav, "End", self.go_to_end, "End")
        nav.addSeparator()
        self._action(nav, "Previous variation", lambda: self.switch_variation(-1), "Up")
        self._action(nav, "Next variation", lambda: self.switch_variation(1), "Down")

        analysis = bar.addMenu("&Analysis")
        self.analyze_action = self._action(analysis, "&Analyze", self.toggle_analysis, "F5", checkable=True)
        self.threat_action = self._action(analysis, "Show &threat", self.toggle_threat, "X", checkable=True)
        self.hint_action = self._action(analysis, "&Hint", self.hint, "H")
        self.review_action = self._action(analysis, "&Review game…", self.start_review, "Ctrl+R")
        analysis.addSeparator()
        self.analysis_engine_menu = analysis.addMenu("Analysis &engine")
        self._fill_analysis_menu()

        tournament = bar.addMenu("&Tournament")
        self.tournament_action = self._action(tournament, "&New tournament…", self.new_tournament, "Ctrl+T")
        self._action(tournament, "Resume a &saved tournament…", self.resume_saved_tournament)
        tournament.addSeparator()
        self.pause_action = self._action(tournament, "&Pause", self.tournament.pause)
        self.resume_action = self._action(tournament, "&Resume", self.tournament.resume)
        self.stop_action = self._action(tournament, "&Stop", self.stop_tournament)

        database = bar.addMenu("&Database")
        self._action(database, "&Open database…", self.open_database, "Ctrl+Shift+O")
        self._action(database, "&Add PGN file…", self.add_database_file)
        self._action(database, "&Close database", self.database.clear)

        engines = bar.addMenu("&Engines")
        self._action(engines, "&Manage engines…", self.manage_engines)

        view = bar.addMenu("&View")
        self.flip_action = self._action(view, "&Flip board", self.flip, "F")
        themes = view.addMenu("&Theme")
        self.theme_group = QActionGroup(self)
        for name, t in THEMES.items():
            a = self._action(themes, t.title, lambda _c=False, n=name: self.set_theme(n), checkable=True)
            a.setData(name)
            self.theme_group.addAction(a)
        boards = view.addMenu("&Board")
        self.board_group = QActionGroup(self)
        for name, t in BOARD_THEMES.items():
            a = self._action(boards, t.title, lambda _c=False, n=name: self.set_board_theme(n), checkable=True)
            a.setData(name)
            self.board_group.addAction(a)
        pieces = view.addMenu("&Piece set")
        group = QActionGroup(self)
        for name, cls in PIECE_SETS.items():
            a = self._action(pieces, cls.title, lambda _c=False, n=name: self.set_piece_set(n), checkable=True)
            a.setData(name)
            group.addAction(a)
        self.piece_actions = group
        self.coords_action = self._action(view, "Show &coordinates", self.toggle_coordinates, checkable=True)
        view.addAction(self.log_dock.toggleViewAction())
        view.addSeparator()
        self.settings_action = self._action(view, "&Settings…", self.open_settings, "Ctrl+,")
        self._check_view_actions()

        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "&Keyboard shortcuts", self.show_shortcuts, "F1")
        self._action(help_menu, "&About", lambda: QMessageBox.about(self, "About ChessGPU GUI", ABOUT))

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("Main")
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._tool_icons: list[tuple[QAction, str]] = []
        for action, name, text in ((self.new_action, "new", "New game"), (self.open_action, "open", "Open"),
                                   (self.save_action, "save", "Save"), (None, "", ""),
                                   (self.flip_action, "flip", "Flip"), (self.analyze_action, "analyze", "Analyze"),
                                   (self.review_action, "review", "Review"), (None, "", ""),
                                   (self.settings_action, "settings", "Settings")):
            if action is None:
                self.toolbar.addSeparator()
                continue
            action.setIconText(text)
            self._tool_icons.append((action, name))
            self.toolbar.addAction(action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self._retheme_icons()

    def _retheme_icons(self) -> None:
        color = current_theme().text
        for action, name in self._tool_icons:
            action.setIcon(icon(name, color))
        self.navigation.retheme()
        for card in self.cards.values():
            card.retheme()

    def _check_view_actions(self) -> None:
        for group, value in ((self.theme_group, self.view.theme), (self.board_group, self.view.board_theme),
                             (self.piece_actions, self.view.piece_set)):
            for a in group.actions():
                a.setChecked(a.data() == value)
        self.coords_action.setChecked(self.view.coordinates)

    def _fill_analysis_menu(self) -> None:
        self.analysis_engine_menu.clear()
        group = QActionGroup(self)
        if self.analysis_profile_name not in [p.name for p in self.profiles] and self.profiles:
            self.analysis_profile_name = self.profiles[0].name
        for p in self.profiles:
            a = self._action(self.analysis_engine_menu, p.name,
                             lambda _c=False, n=p.name: self.set_analysis_profile(n), checkable=True)
            a.setChecked(p.name == self.analysis_profile_name)
            group.addAction(a)
        self.analysis_group = group

    # --- Look and feel ---------------------------------------------------------------

    def _apply_view(self, v: ViewSettings) -> None:
        """Applies settings to the widgets (not to the settings file)."""
        b = self.board
        b.set_board_theme(BOARD_THEMES[v.board_theme])
        b.show_coordinates = v.coordinates
        b.show_legal = v.show_legal
        b.highlight_last = v.highlight_last
        b.auto_queen = v.auto_queen
        b.animation_ms = v.animation_ms
        b.update()
        if self.piece_set_name != v.piece_set or b.piece_set.name != v.piece_set:
            self.piece_set_name = v.piece_set
            piece_set = make_piece_set(v.piece_set)
            b.set_piece_set(piece_set)
            for card in self.cards.values():
                card.set_piece_set(piece_set)
        self.move_tree.figurines = v.figurines
        self.explorer_panel.figurines = v.figurines
        for out in self.outputs.values():
            out.figurines = v.figurines
        self.sounds.enabled = v.sounds
        self.sounds.volume = v.volume / 100
        if not v.premoves:
            b.cancel_premoves()

    def set_view(self, v: ViewSettings) -> None:
        """Applies and stores new settings (theme included)."""
        old_theme = self.view.theme
        self.view = v
        v.save(self.settings)
        if v.theme != old_theme:
            app = QApplication.instance()
            if app is not None:
                apply_theme(app, v.theme)
            self._retheme_icons()
        self._apply_view(v)
        self._check_view_actions()
        self._refresh()
        for w in (self.eval_bar, self.graph):
            w.update()

    def _replace_view(self, **changes) -> None:
        values = dict(self.view.__dict__)
        values.update(changes)
        self.set_view(ViewSettings(**values))

    def set_theme(self, name: str) -> None:
        self._replace_view(theme=name)

    def set_board_theme(self, name: str) -> None:
        self._replace_view(board_theme=name)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.view, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_view(dialog.result_settings())

    def show_shortcuts(self) -> None:
        actions = [a for a in self.findChildren(QAction) if not a.shortcut().isEmpty() and a.text()]
        table = QTableWidget(len(actions), 2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        for row, a in enumerate(sorted(actions, key=lambda a: a.text().replace("&", ""))):
            table.setItem(row, 0, QTableWidgetItem(a.text().replace("&", "").rstrip("…")))
            table.setItem(row, 1, QTableWidgetItem(a.shortcut().toString(QKeySequence.SequenceFormat.NativeText)))
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        layout.addWidget(table)
        hint = QLabel("On the board: right-click a square for a circle, right-drag for an arrow "
                      "(Shift red, Ctrl blue, Alt yellow); a left click clears them.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(460, 520)
        self.shortcuts_dialog = dialog
        dialog.exec()

    # --- Views ----------------------------------------------------------------------

    @property
    def viewed(self) -> Position:
        return self.view_node.position

    @property
    def view_ply(self) -> int:
        return self.view_node.ply

    def go_to(self, ply: int) -> None:
        """Shows the position after `ply` moves of the line shown (the path
        to the shown position, then its main continuation)."""
        line = self.view_node.line()
        self.go_to_node(line[max(0, min(len(line) - 1, ply))])

    def go_to_end(self) -> None:
        self.go_to_node(self.view_node.end())

    def go_to_node(self, node: Node, sound: bool = True) -> None:
        """Shows `node`'s position; a step forward plays its move's sound."""
        if not self.game.record.contains(node):
            return
        step = node.parent is self.view_node
        self.view_node = node
        self._follow = node is self.game.record.end
        self._refresh()
        if step and sound:
            self._play_move_sound(node)

    def switch_variation(self, delta: int) -> None:
        """Shows the previous (-1) or next (+1) move played from the position
        before the one shown."""
        siblings = self.view_node.siblings()
        i = next(k for k, n in enumerate(siblings) if n is self.view_node)
        if 0 <= i + delta < len(siblings):
            self.go_to_node(siblings[i + delta], sound=False)

    def can_move(self) -> bool:
        g = self.game
        if g.status == g.RUNNING:
            return g.human_to_move and self.view_node is g.record.end
        if g.status == g.PREPARING:
            return False
        return bool(self.viewed.legal_moves())

    def _human_color(self) -> int | None:
        """The human's side in a game against an engine, else None."""
        g = self.game
        humans = [c for c in (WHITE, BLACK) if g.profiles[c] is None]
        return humans[0] if len(humans) == 1 else None

    def _refresh(self) -> None:
        g = self.game
        record = g.record
        if not record.contains(self.view_node):
            self.view_node = record.end
        node = self.view_node
        self.board.set_position(node.position, node.move)
        self.board.set_shapes(node.arrows, node.circles)
        self.board.set_interactive(self.can_move())
        self._update_premoves()
        self.move_tree.can_edit = g.status not in (g.RUNNING, g.PREPARING)
        self.move_tree.set_tree(record, node)
        self.comment_box.set_node(node)
        self._update_review_view()
        self.explorer_panel.show_position(self.database, node.position, node.ply)
        if self.database_panel.here.isChecked():
            self._search_database()
        line = node.line()
        self.graph.set_line(line, node.ply)
        self._mark_graph(line)
        for card in self.cards.values():
            card.set_position(node.position)
        if not self.analysis.running:
            self.board.set_arrows([])
            self.eval_bar.set_score(self._score_at(node))
        self._update_actions()
        self._update_panels()
        self._update_title()
        self._update_analysis()

    def _update_premoves(self) -> None:
        g = self.game
        human = self._human_color()
        waiting = (g.status == g.RUNNING and human is not None and g.record.position.turn != human
                   and self.view_node is g.record.end)
        self.board.set_premove_color(human if waiting and self.view.premoves else None)

    @staticmethod
    def _score_at(node: Node):
        """White's-view score of `node`'s position, or of the closest one
        before it that has one (review or engine report)."""
        n: Node | None = node
        while n is not None:
            score = node_score(n)
            if score is not None:
                return score
            n = n.parent
        return None

    def _update_actions(self) -> None:
        g = self.game
        human_game = g.profiles[WHITE] is None or g.profiles[BLACK] is None
        self.takeback_action.setEnabled(g.status in (g.RUNNING, g.OVER) and human_game and bool(g.record.moves))
        self.move_now_action.setEnabled(g.thinking_color is not None)
        self.resign_action.setEnabled(g.status == g.RUNNING and g.human_to_move)
        self.abort_action.setEnabled(g.status in (g.RUNNING, g.PREPARING))
        self.analyze_action.setEnabled(g.status not in (g.RUNNING, g.PREPARING) or self.analysis.running)
        self.threat_action.setEnabled(g.status not in (g.RUNNING, g.PREPARING))
        self.review_action.setEnabled(g.status not in (g.RUNNING, g.PREPARING) and not self.review.running)
        t = self.tournament
        self.tournament_action.setEnabled(not t.active and g.status not in (g.RUNNING, g.PREPARING))
        self.pause_action.setEnabled(t.state == t.RUNNING)
        self.resume_action.setEnabled(t.state == t.PAUSED)
        self.stop_action.setEnabled(t.active)
        self.hint_action.setEnabled(g.status == g.RUNNING and g.human_to_move)

    def _update_panels(self) -> None:
        g = self.game
        analysing = self.analysis.running
        for color, key in ((WHITE, "white"), (BLACK, "black")):
            profile = g.profiles[color]
            self.outputs[key].setVisible(profile is not None and not analysing)
            if profile is not None:
                thinking = " (thinking)" if g.thinking_color == color else ""
                self.outputs[key].title.setText(f"{COLOR_NAMES[color]}: {profile.name}{thinking}")
        self.outputs["analysis"].setVisible(analysing)
        for card in self.cards.values():
            card.refresh()

    def _tick(self) -> None:
        """Clocks (every 100 ms) and the human's low-time warning."""
        for card in self.cards.values():
            card.refresh()
        g = self.game
        human = self._human_color()
        if (g.status == g.RUNNING and g.tc.clocked and human is not None and g.clock.running == human
                and human not in self._warned and g.clock.left(human) < LOW_TIME_WARNING_MS):
            self._warned.add(human)
            self.sounds.play("low_time")

    def _update_title(self) -> None:
        r = self.game.record
        names = f"{r.tags.get('White', '?')} – {r.tags.get('Black', '?')}" if r.tags else "analysis board"
        result = (f"  {r.result}" if r.result != "*" else "") + (f" ({r.reason})" if r.reason else "")
        self.setWindowTitle(f"ChessGPU — {names}{result}")

    def show_message(self, text: str) -> None:
        self.status.setText(text)

    def _log(self, name: str, direction: str, line: str) -> None:
        self.log_view.appendPlainText(f"[{name}] {direction} {line}")

    # --- Sounds -------------------------------------------------------------------------

    def _play_move_sound(self, node: Node) -> None:
        """The sound of the move that led to `node`."""
        if node.parent is not None:
            self.sounds.play(move_sound(node.parent.position, node.move, node.position))

    def _game_sounds(self) -> None:
        """A game's start and moves (the window plays explored moves itself)."""
        g = self.game
        moves = len(g.record.moves)
        if self._seen_status == g.PREPARING and g.status == g.RUNNING:
            self._warned = set()
            self.sounds.play("start")
        elif moves == self._seen_moves + 1 and g.status == g.RUNNING:
            self._play_move_sound(g.record.end)
        self._seen_moves = moves
        self._seen_status = g.status

    # --- Game events ------------------------------------------------------------------

    def _game_changed(self) -> None:
        record = self.game.record
        if self.review.running and record is not self._review_record:
            self.review.cancel()  # another game: the review was for the one before
        if self._follow or not record.contains(self.view_node):
            self.view_node = record.end
        self._refresh()
        self._game_sounds()
        self._play_premove()

    def _play_premove(self) -> None:
        g = self.game
        if not g.human_to_move or not self.board.premoves:
            return
        move = self.board.take_premove()
        if move is not None and g.record.position.is_legal(move):
            QTimer.singleShot(0, lambda m=move: g.human_move(m))
        elif move is not None:
            self.show_message(f"Premove {move.uci()} cancelled: not legal now")

    def _engine_info(self, color: int, info) -> None:
        key = "white" if color == WHITE else "black"
        self.outputs[key].show_info(info, self.game.record.position)
        if self._follow and not self.analysis.running and self.outputs[key].score is not None:
            self.eval_bar.set_score(self.outputs[key].score)

    def _game_over(self, result: str, reason: str) -> None:
        self.board.cancel_premoves()
        self.sounds.play("end")
        self.show_message(f"{result}: {reason}" if result != "*" else reason)

    def _board_move(self, move: Move) -> None:
        g = self.game
        if g.status == g.RUNNING:
            g.human_move(move)
            return
        node = g.explore(self.view_node, move)
        if node is not None:
            self.go_to_node(node, sound=False)
            self._play_move_sound(node)

    def _shapes_drawn(self) -> None:
        """Arrows and circles belong to the position they were drawn on."""
        self.view_node.set_shapes(self.board.shape_arrows, self.board.shape_circles)

    def _tree_action(self, name: str, node: Node) -> None:
        """A move tree context menu choice for `node`."""
        g = self.game
        record = g.record
        if not record.contains(node):
            return
        if name.startswith("nag:"):
            node.nags = toggle_nag(node.nags, int(name[4:]))
        elif name == "clear_nags":
            node.nags = []
        elif name == "comment":
            self.go_to_node(node, sound=False)
            self.comment_box.setFocus()
            return
        elif name == "copy_line":
            QGuiApplication.clipboard().setText(line_text(node, self.view.figurines))
            self.show_message("Line copied")
            return
        elif name == "copy_fen":
            QGuiApplication.clipboard().setText(node.position.fen())
            self.show_message("FEN copied")
            return
        elif name in ("promote", "make_main", "delete"):
            if g.status in (g.RUNNING, g.PREPARING):
                self.show_message("Finish or abort the game first.")
                return
            if name == "delete" and any(n is node for n in self.view_node.path()):
                self.view_node, self._follow = node.parent, False  # the shown position goes: show the one before
            change = {"promote": record.promote, "make_main": record.make_main, "delete": record.delete}[name]
            g.edit(change, node)
            return
        self.move_tree.refresh()

    # --- Analysis ---------------------------------------------------------------------

    def _analysis_profile(self) -> EngineProfile | None:
        for p in self.profiles:
            if p.name == self.analysis_profile_name:
                return p
        return self.profiles[0] if self.profiles else None

    def set_analysis_profile(self, name: str) -> None:
        self.analysis_profile_name = name
        self.settings.setValue("analysis_engine", name)
        if self.analysis.running:
            self.analysis.stop()
            self.start_analysis()

    def toggle_analysis(self, on: bool) -> None:
        if on:
            self.start_analysis()
        else:
            self.stop_analysis()

    def start_analysis(self) -> None:
        if self.tournament.state == self.tournament.RUNNING:
            self.analyze_action.setChecked(False)
            self._tournament_busy()
            return
        g = self.game
        profile = self._analysis_profile()
        if g.status in (g.RUNNING, g.PREPARING) or profile is None:
            self.analyze_action.setChecked(False)
            self.show_message("Analysis is off while a game runs." if profile else "No engine to analyze with.")
            return
        self.analysis.set_profile(profile)
        self.outputs["analysis"].clear(self._analysis_title())
        self.analyze_action.setChecked(True)
        self._analysed = None
        self._update_analysis(force=True)
        self._refresh()

    def stop_analysis(self) -> None:
        self.analysis.stop()
        self._analysed = None
        self.analyze_action.setChecked(False)
        self.board.set_arrows([])
        self._refresh()

    def _analysis_title(self) -> str:
        profile = self._analysis_profile()
        name = profile.name if profile else ""
        return f"Threat: {name} (what the opponent would play)" if self.threat_mode else f"Analysis: {name}"

    def _analysis_target(self) -> tuple[Position, tuple] | None:
        """(position analysed, (start FEN, moves) for the engine): the shown
        position, or in threat mode the one after a pass (None in check)."""
        node = self.view_node
        if self.threat_mode:
            try:
                after_pass = node.position.null_move()
            except IllegalMoveError:
                return None
            return after_pass, (after_pass.fen(), ())
        return node.position, (self.game.record.start.fen(), tuple(node.uci_moves()))

    def _update_analysis(self, force: bool = False) -> None:
        if not (self.analyze_action.isChecked() or force):
            return
        found = self._analysis_target()
        target = found[1] if found else ("in check", ())
        if target == self._analysed and self.analysis.running:
            return
        self._analysed = target
        self.outputs["analysis"].clear(self._analysis_title())
        self.board.set_arrows([])
        if found is None:
            self.analysis.stop()
            self.outputs["analysis"].title.setText("Threat: none to show while in check")
            return
        self._analysis_root = found[0]
        if found[0].legal_moves():
            self.analysis.analyze(target[0], list(target[1]))
        else:
            self.analysis.stop()

    def toggle_threat(self, on: bool) -> None:
        """Threat mode: the analysis shows what the opponent would play if
        it were their move (red arrows)."""
        self.threat_mode = bool(on)
        self.threat_action.setChecked(self.threat_mode)
        if self.analysis.running or self.analyze_action.isChecked():
            self._analysed = None
            self._update_analysis(force=True)
        elif self.threat_mode:
            self.start_analysis()
            if not self.analyze_action.isChecked():  # refused (a game runs, no engine)
                self.threat_mode = False
                self.threat_action.setChecked(False)

    def set_analysis_lines(self, n: int) -> None:
        n = max(1, min(MAX_ANALYSIS_LINES, int(n)))
        self.analysis_lines = n
        self.settings.setValue("analysis_lines", n)
        if self.lines_box.value() != n:
            self.lines_box.setValue(n)
        self.outputs["analysis"].set_max_lines(n)
        self.analysis.set_lines(n)
        if self.analysis.running and self.analysis.engine is not None and self.analysis.engine.running and \
                n > 1 and not self.analysis.has_multipv():
            self.show_message("This engine has no MultiPV option: one line only.")

    def _pv_clicked(self, rank: int, count: int) -> None:
        """Plays an analysis line up to its `count`th move (as a variation)."""
        out = self.outputs["analysis"]
        if self.threat_mode or rank not in out.lines or out.root is None or out.root != self.viewed:
            return
        g = self.game
        if g.status in (g.RUNNING, g.PREPARING):
            return
        moves, pos = [], out.root
        for text in out.lines[rank][1][:count]:
            try:
                move = pos.parse_uci(text)
            except IllegalMoveError:
                break
            moves.append(move)
            pos = pos.push(move)
        node = g.edit(g.record.add_line, self.view_node, moves) if moves else None
        if node is not None:
            self.go_to_node(node, sound=False)
            self._play_move_sound(node)

    def hint(self) -> None:
        """A short search by the analysis engine for the human to move in a game."""
        g = self.game
        profile = self._analysis_profile()
        if not (g.status == g.RUNNING and g.human_to_move):
            self.show_message("Hints are for your move in a game (outside games, use Analyze).")
            return
        if profile is None:
            self.show_message("No engine to give hints with.")
            return
        self.analysis.set_profile(profile)
        self._hint_for = g.record.end
        if self.analysis.hint(g.record.start.fen(), g.record.uci_moves(), HINT_MS):
            self.show_message("Looking for a hint…")

    def _hint_ready(self, best, info) -> None:
        g = self.game
        node, self._hint_for = self._hint_for, None
        if best is None or node is None or node is not g.record.end or not g.human_to_move:
            return  # the game moved on
        try:
            move = node.position.parse_uci(best)
        except IllegalMoveError:
            return
        self.board.set_arrows([move])
        score = white_score(info.score, node.position.turn) if info is not None and info.score is not None else None
        self.show_message(f"Hint: {node.position.san(move)}" + (f" ({score.text()})" if score else ""))

    def _analysis_running(self, running: bool) -> None:
        """Keeps the Analyze action in step when analysis stops by itself
        (e.g. its engine failed)."""
        if not running and self.analyze_action.isChecked() and self.analysis.engine is not None and \
                not self.analysis.engine.running:
            self.analyze_action.setChecked(False)
            self._update_panels()

    def _analysis_info(self, info) -> None:
        out = self.outputs["analysis"]
        root = self._analysis_root or self.viewed
        out.show_info(info, root)
        if out.score is not None and not self.threat_mode:  # (a threat's score is the opponent's to move)
            self.eval_bar.set_score(out.score)
        if info.pv:
            arrows = []
            for rank in sorted(out.lines):
                try:
                    arrows.append(root.parse_uci(out.lines[rank][1][0]))
                except (ValueError, IndexError):
                    continue
            base = THREAT_ARROW if self.threat_mode else ENGINE_ARROW
            self.board.set_arrows(arrows, line_arrow_colors(base, len(arrows)))

    # --- Tournaments -------------------------------------------------------------------

    def _profile_named(self, name: str) -> EngineProfile | None:
        return next((p for p in self.profiles if p.name == name), None)

    def _tournament_busy(self) -> bool:
        """A tournament game is being played (says so if it is)."""
        if self.tournament.state == self.tournament.RUNNING:
            self.show_message("A tournament is under way: pause or stop it first.")
            return True
        return False

    def new_tournament(self) -> None:
        g = self.game
        if self.tournament.active or g.status in (g.RUNNING, g.PREPARING):
            self.show_message("Finish the game or tournament first.")
            return
        last = self.settings.value("tournament", {})
        folder = pathlib.Path(self.settings.fileName()).parent / "tournaments"
        dialog = TournamentDialog(self.profiles, last if isinstance(last, dict) else {}, str(folder), self)
        self.tournament_dialog = dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.setValue("tournament", dialog.settings())
        self.start_tournament(dialog.tournament(), dialog.pgn.text().strip())

    def start_tournament(self, settings, pgn_path: str, results=None) -> bool:
        if self.analysis.running:
            self.stop_analysis()
        if self.review.running:
            self.review.cancel()
        try:
            pathlib.Path(pgn_path).parent.mkdir(parents=True, exist_ok=True)
            self.tournament.start(settings, pgn_path, results)
        except (OSError, ValueError, RuntimeError) as e:
            QMessageBox.warning(self, "Tournament", f"Couldn't start the tournament:\n{e}")
            return False
        self.tabs.setCurrentWidget(self.tournament_panel)
        return True

    def resume_saved_tournament(self) -> None:
        if self.tournament.active:
            self.show_message("A tournament is under way.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Resume a tournament", str(self.settings.value("pgn_dir", "")),
                                              "Tournament state (*.json)")
        if path:
            self.resume_tournament_from(path)

    def resume_tournament_from(self, path: str) -> bool:
        try:
            settings, pgn, results = TournamentController.load_state(path)
        except ValueError as e:
            QMessageBox.warning(self, "Tournament", str(e))
            return False
        return self.start_tournament(settings, pgn, results)

    def stop_tournament(self) -> None:
        if self.tournament.active:
            self.tournament.stop()
            self.show_message("Tournament stopped: resume it later from its .json file.")

    def _tournament_game_started(self, pairing) -> None:
        self.game.book, self.game.book_plies = None, 0  # (tournaments take their openings from files)
        for out in self.outputs.values():
            out.clear()
        self.board.cancel_premoves()
        self.board.clear_shapes()
        self._follow = True
        self._warned = set()

    def _tournament_changed(self) -> None:
        t = self.tournament
        s = t.settings
        if s is None:
            return
        total, done = len(t.pairings), len(t.pairings) - len(t.remaining())
        if t.state == t.RUNNING and t.current is not None:
            p = t.current
            text = f"{s.name}: game {p.number} of {total}, {p.white} – {p.black}"
            if t.pausing:
                text += " (pausing after this game)"
        elif t.state == t.PAUSED:
            text = f"{s.name}: paused after {done} of {total} games"
        elif t.state == t.RUNNING:
            text = f"{s.name}: {done} of {total} games"
        else:
            text = f"{s.name}: {'finished' if done == total else 'stopped'}, {done} of {total} games"
        self.tournament_panel.set_state(t.state, text)
        self.tournament_panel.show_results(s.engines, t.standings(), t.results)
        self._update_actions()

    def _tournament_finished(self, complete: bool) -> None:
        if complete:
            table = self.tournament.standings()
            self.show_message("Tournament over: " + ", ".join(f"{x.engine} {x.points:g}" for x in table))

    def show_tournament_game(self, number: int) -> None:
        """Opens a finished tournament game from its PGN file."""
        if self._tournament_busy() or self.tournament.pgn_path is None:
            return
        try:
            games = read_pgn(self.tournament.pgn_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, PgnError) as e:
            self.show_message(f"Couldn't read the tournament's games: {e}")
            return
        for g in games:
            if g.tags.get("Round") == str(number) and g.tags.get("Event") == self.tournament.settings.name:
                self._load(GameRecord.from_pgn(g))
                self.tabs.setCurrentIndex(0)
                return
        self.show_message(f"Game {number} isn't in {self.tournament.pgn_path}.")

    # --- Databases ---------------------------------------------------------------------

    def open_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a database of games",
                                              str(self.settings.value("database_dir", "")),
                                              "PGN files (*.pgn);;All files (*)")
        if path:
            self.open_database_from(path)

    def add_database_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add games to the database",
                                              str(self.settings.value("database_dir", "")),
                                              "PGN files (*.pgn);;All files (*)")
        if path:
            self.open_database_from(path, add=True)

    def open_database_from(self, path: str, add: bool = False, wait: bool = False) -> bool:
        """Opens (or adds) a PGN file as the database, then indexes its
        positions in the background (`wait`: before returning)."""
        if not add:
            self.database.clear()
        try:
            count = self.database.add_file(path)
        except OSError as e:
            QMessageBox.warning(self, "Database", f"Couldn't read {path}:\n{e}")
            return False
        self.settings.setValue("database_dir", str(pathlib.Path(path).parent))
        self.show_message(f"{count} games from {path}; indexing their positions…")
        self.tabs.setCurrentWidget(self.database_panel)
        self.database.build_index(wait=wait)
        self._database_changed()
        return True

    def _database_changed(self) -> None:
        if not self.database.building:
            self.database_panel.show_progress(1, 1)  # (hides the bar)
        self._search_database()
        self.explorer_panel.show_position(self.database, self.view_node.position, self.view_node.ply)

    def _search_database(self) -> None:
        text, result, here = self.database_panel.filters()
        games = self.database.search(text, result, self.view_node.position if here else None)
        self.database_panel.show_games(self.database, games)

    def open_database_game(self, number: int) -> None:
        game = self.database.game(number)
        if game is None or self._tournament_busy():
            return
        g = self.game
        if g.status in (g.RUNNING, g.PREPARING):
            self.show_message("Finish or abort the game first.")
            return
        try:
            record = game.record()
        except PgnError as e:
            QMessageBox.warning(self, "Database", f"Couldn't read game {number}:\n{e}")
            return
        self._load(record)
        self.tabs.setCurrentIndex(0)
        self.show_message(f"Game {number} of {len(self.database.games)} from {self.database.name}")

    def _explorer_move(self, move: Move) -> None:
        """An explorer move: played as if on the board."""
        if self.board.interactive and self.viewed.is_legal(move):
            self._board_move(move)

    # --- Game review --------------------------------------------------------------------

    def start_review(self) -> None:
        """Asks how to review the game, then reviews it."""
        if self._tournament_busy():
            return
        g = self.game
        if g.status in (g.RUNNING, g.PREPARING):
            self.show_message("Finish or abort the game first.")
            return
        if not g.record.moves:
            self.show_message("No moves to review.")
            return
        if not self.profiles:
            self.show_message("No engine to review with.")
            return
        last = self.settings.value("review", {})
        dialog = ReviewDialog(self.profiles, self.analysis_profile_name, len(g.record.moves) + 1,
                              last if isinstance(last, dict) else {}, self)
        self.review_dialog = dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.setValue("review", dialog.settings())
        self.run_review(*dialog.choice())

    def run_review(self, engine: str, movetime_ms: int | None = 1000, depth: int | None = None,
                   annotate_game: bool = True) -> bool:
        profile = next((p for p in self.profiles if p.name == engine), None)
        g = self.game
        if profile is None or g.status in (g.RUNNING, g.PREPARING) or not g.record.moves:
            return False
        if self.analysis.running:
            self.stop_analysis()
        self._review_record = g.record
        self._review_annotate = annotate_game
        self.review.start(g.record, profile, movetime_ms, depth)
        self.tabs.setCurrentWidget(self.review_panel)
        self.review_panel.set_running(True, 0, len(g.record.moves) + 1)
        self._update_actions()
        return True

    def _review_progress(self, done: int, total: int) -> None:
        self.review_panel.set_running(True, done, total)
        self._update_review_view()
        line = self.view_node.line()
        self.graph.set_line(line, self.view_node.ply)
        self._mark_graph(line)

    def _review_finished(self, complete: bool) -> None:
        record, self._review_record = self._review_record, None
        self.review_panel.set_running(False)
        if record is not None and record is self.game.record:
            judgements = judge(record)
            if complete and self._review_annotate:
                annotate(record, judgements)
            sides = summarize(judgements)
            if complete:
                self.show_message("Review done: " + ", ".join(
                    f"{name} {s.accuracy:.0f}%" for name, s in (("White", sides[WHITE]), ("Black", sides[BLACK]))
                    if s.moves))
            else:
                self.show_message("Review stopped: the moves reviewed so far are judged.")
            self._review_signature = []
            self._refresh()
        self._update_actions()

    def _update_review_view(self) -> None:
        """The review panel, from the reviews the game's positions carry."""
        record = self.game.record
        judgements = judge(record) if record.root.review is not None or any(
            n.review is not None for n in record.moves) else []
        signature = [(id(j.node), j.kind, round(j.loss, 4)) for j in judgements]
        if signature != self._review_signature or not judgements:
            self._review_signature = signature
            self.judgements = judgements
            if judgements:
                self.review_panel.show_results(record, judgements, summarize(judgements), self.view.figurines)
            elif not self.review.running:
                self.review_panel.clear()

    def _mark_graph(self, line: list[Node]) -> None:
        on_line = {id(n) for n in line}
        self.graph.set_marks({j.node.ply: kind_color(j.kind) for j in self.judgements
                              if j.kind is not None and id(j.node) in on_line})

    # --- Commands ---------------------------------------------------------------------

    def new_game(self) -> None:
        if self._tournament_busy():
            return
        last = self.settings.value("new_game", {})
        book = self.database.name if self.database.index is not None else ""
        dialog = NewGameDialog(self.profiles, self.viewed, last if isinstance(last, dict) else {}, self, book)
        self.new_game_dialog = dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.setValue("new_game", dialog.settings())
        choice = dialog.choice()
        self.start_game(choice.white, choice.black, choice.tc, choice.start, choice.book_plies)

    def start_game(self, white, black, tc, start=None, book_plies: int = 0) -> None:
        if self._tournament_busy():
            return
        self.game.book = self.database.index if book_plies else None
        self.game.book_plies = book_plies
        if self.analysis.running:
            self.stop_analysis()
        if self.review.running:
            self.review.cancel()
        for out in self.outputs.values():
            out.clear()
        self.board.cancel_premoves()
        self.board.clear_shapes()
        self._follow = True
        self._warned = set()
        self.game.new_game(white, black, tc, start)
        self.show_message(f"New game: {self.game.player_name(WHITE)} – {self.game.player_name(BLACK)}, "
                          f"{tc.text()}")

    def takeback(self) -> None:
        if self._tournament_busy():
            return
        self._follow = True
        self.board.cancel_premoves()
        if not self.game.takeback():
            self.show_message("Nothing to take back.")

    def resign(self) -> None:
        self.game.resign()

    def save_pgn(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save game", str(self.settings.value("pgn_dir", "")),
                                              "PGN files (*.pgn);;All files (*)")
        if path:
            self.save_pgn_to(path)

    def save_pgn_to(self, path: str) -> None:
        pathlib.Path(path).write_text(write_pgn(self.game.record.to_pgn()), encoding="utf-8")
        self.settings.setValue("pgn_dir", str(pathlib.Path(path).parent))
        self.show_message(f"Saved {path}")

    def open_pgn(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open games", str(self.settings.value("pgn_dir", "")),
                                              "PGN files (*.pgn);;All files (*)")
        if path:
            self.open_pgn_from(path)

    def open_pgn_from(self, path: str, index: int | None = None) -> bool:
        if self._tournament_busy():
            return False
        try:
            games = read_pgn(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
        except (OSError, PgnError) as e:
            QMessageBox.warning(self, "Open games", f"Couldn't read {path}:\n{e}")
            return False
        if not games:
            self.show_message(f"No games in {path}")
            return False
        if index is None:
            index = 0
            if len(games) > 1:
                labels = [f"{i + 1}. {g.tags.get('White', '?')} – {g.tags.get('Black', '?')}  {g.result}"
                          for i, g in enumerate(games)]
                label, ok = QInputDialog.getItem(self, "Open games", f"{len(games)} games:", labels, 0, False)
                if not ok:
                    return False
                index = labels.index(label)
        self._load(GameRecord.from_pgn(games[index]))
        self.settings.setValue("pgn_dir", str(pathlib.Path(path).parent))
        self.show_message(f"Opened game {index + 1} of {len(games)} from {path}")
        return True

    def _load(self, record: GameRecord, at_end: bool = True) -> None:
        if self.analysis.running:
            self.analysis.stop()
        self.board.cancel_premoves()
        self.board.clear_shapes()
        self.game.load(record)
        self.go_to_node(record.end if at_end else record.root, sound=False)
        if self.analyze_action.isChecked():
            self.start_analysis()

    def copy_fen(self) -> None:
        QGuiApplication.clipboard().setText(self.viewed.fen())
        self.show_message("FEN copied")

    def paste_fen(self) -> None:
        self.load_fen(QGuiApplication.clipboard().text().strip())

    def load_fen(self, fen: str) -> bool:
        if self._tournament_busy():
            return False
        try:
            pos = Position.from_fen(fen)
        except FenError as e:
            self.show_message(f"Not a valid FEN: {e}")
            return False
        self._load(GameRecord(pos), at_end=False)
        return True

    def edit_position(self) -> None:
        if self._tournament_busy():
            return
        if self.game.status in (self.game.RUNNING, self.game.PREPARING):
            self.show_message("Finish or abort the game first.")
            return
        editor = BoardEditor(self.viewed, self.board.piece_set, self.board.flipped, self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self._load(GameRecord(editor.result_position()), at_end=False)

    def manage_engines(self) -> None:
        dialog = ProfilesDialog(self.profiles, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.profiles = dialog.profiles
            self.settings.setValue("profiles", profiles_to_json(self.profiles))
            self._fill_analysis_menu()

    def flip(self) -> None:
        self.board.set_flipped(not self.board.flipped)
        self.eval_bar.set_flipped(self.board.flipped)
        self._place_cards()
        self.settings.setValue("flipped", "true" if self.board.flipped else "false")

    def set_piece_set(self, name: str) -> None:
        self._replace_view(piece_set=name)

    def toggle_coordinates(self, on: bool) -> None:
        self._replace_view(coordinates=bool(on))

    def closeEvent(self, event) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("profiles", profiles_to_json(self.profiles))
        self.settings.sync()
        self._card_timer.stop()
        if self.review.running:
            self.review.cancel()
        if self.tournament.active:
            self.tournament.stop()  # (its state is saved: it can be resumed)
        self.analysis.shutdown()
        self.game.shutdown()
        super().closeEvent(event)
