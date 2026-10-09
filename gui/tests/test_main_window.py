"""The main window end to end (offscreen), with the scripted fake engine."""

import pathlib
import sys

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from chessgpu_gui.game import TimeControl
from chessgpu_gui.main_window import MainWindow
from chessgpu_gui.pgn import read_pgn
from chessgpu_gui.pieces import GlyphPieceSet, SvgPieceSet, VectorPieceSet
from chessgpu_gui.profiles import EngineProfile, profiles_to_json
from chessgpu_gui.rules import BLACK, START_FEN, WHITE, Move, Position, parse_square
from chessgpu_gui.theme import ENGINE_ARROW, THREAT_ARROW
from qt_helpers import wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))
DATA = pathlib.Path(__file__).resolve().parent / "data"


def fake(*flags, name="fake"):
    return EngineProfile(name, sys.executable, [FAKE, *flags])


@pytest.fixture
def window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json([fake(), fake("--wait-for-stop", name="thinker")]))
    w = MainWindow(settings)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    yield w
    w.close()


def click(board, name):
    p = board.square_rect(parse_square(name)).center()
    for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                          (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        event = QMouseEvent(kind, QPointF(p), QPointF(board.mapToGlobal(QPoint(int(p.x()), int(p.y())))),
                            Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(board, event)


def move(board, uci):
    click(board, uci[:2])
    click(board, uci[2:4])


def click_move(view, node):
    """Clicks a move in the move tree (as the mouse would)."""
    from PySide6.QtTest import QTest

    i = next(k for k, n in enumerate(view._nodes) if n is node)
    block = view.document().begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            frag = it.fragment()
            if frag.isValid() and frag.charFormat().anchorHref() == f"n:{i}":
                cursor = view.textCursor()
                cursor.setPosition(frag.position() + 1)
                view.setTextCursor(cursor)
                view.ensureCursorVisible()
                point = view.cursorRect(cursor).center()
                QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
                return
            it += 1
        block = block.next()
    raise AssertionError(f"{node} isn't shown")


def test_profiles_come_from_settings(window):
    assert [p.name for p in window.profiles] == ["fake", "thinker"]
    assert window.analysis_profile_name == "fake"


def test_play_against_the_engine_with_the_mouse(window):
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    assert wait_until(lambda: window.game.status == window.game.RUNNING)
    assert window.board.interactive and window.outputs["black"].isVisible() and not window.outputs["white"].isVisible()
    move(window.board, "e2e4")
    assert wait_until(lambda: len(window.game.record.moves) == 2)
    assert window.view_ply == 2 and window.board.position == window.game.record.position
    assert window.board.last_move == Move.from_uci("a7a5")
    assert [text for _, text in window.move_tree.anchors()] == ["e4", "a5"]
    assert window.outputs["black"].fields["Depth"].text() == "2"
    # Browsing back during the game: the board shows the past and takes no moves.
    window.go_to(0)
    assert window.board.position == Position.start() and not window.board.interactive
    move(window.board, "d2d4")
    assert len(window.game.record.moves) == 2
    window.go_to(2)
    assert window.board.interactive
    window.takeback()
    assert len(window.game.record.moves) == 0 and window.view_ply == 0


def test_engine_game_and_pgn_round_trip(window, tmp_path):
    window.start_game(window.profiles[0], window.profiles[0], TimeControl(depth=1))
    assert wait_until(lambda: window.game.status == window.game.OVER, 60_000)
    record = window.game.record
    assert record.result in ("1-0", "0-1", "1/2-1/2") and record.reason in window.windowTitle()
    path = tmp_path / "game.pgn"
    window.save_pgn_to(str(path))
    game = read_pgn(path.read_text())[0]
    assert game.moves == [m.move for m in record.moves] and game.result == record.result
    assert game.tags["White"] == "fake" and game.tags["Termination"] == "normal"
    window.load_fen(START_FEN)
    assert window.game.record.moves == []
    assert window.open_pgn_from(str(path))
    assert [m.move for m in window.game.record.moves] == game.moves
    assert window.view_ply == len(game.moves) and window.game.record.reason == record.reason


def test_open_pgn_with_several_games_and_navigation(window):
    assert window.open_pgn_from(str(DATA / "fastchess_sample.pgn"), index=2)
    record = window.game.record
    assert record.tags["White"] == "thread" and len(record.moves) == 97
    assert window.cards[WHITE].name.text() == "thread"
    window.go_to(0)
    assert window.board.position == record.start
    window.go_to(-5)
    assert window.view_ply == 0
    window.go_to(10_000)
    assert window.view_ply == 97
    click_move(window.move_tree, window.game.record.moves[1])
    assert window.view_ply == 2 and window.view_node is window.game.record.moves[1]
    assert window.eval_bar.score is not None  # the evals came from the PGN comments


def test_analysis_follows_the_viewed_position(window):
    window.open_pgn_from(str(DATA / "fastchess_sample.pgn"), index=0)
    window.set_analysis_profile("thinker")
    window.go_to(4)
    window.start_analysis()
    assert window.analysis.running and window.outputs["analysis"].isVisible()
    assert wait_until(lambda: window.outputs["analysis"].best is not None)
    first = window.outputs["analysis"].best
    assert window.board.arrows and window.board.arrows[0].uci() == first
    window.go_to(5)
    assert wait_until(lambda: window.outputs["analysis"].best not in (None, first))
    # A game can't start analysis-free... and starting one stops the analysis.
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    assert not window.analysis.running and not window.analyze_action.isChecked()
    window.start_analysis()
    assert not window.analysis.running  # refused while the game runs


def test_explore_after_a_game(window):
    window.open_pgn_from(str(DATA / "fastchess_sample.pgn"), index=2)
    record = window.game.record
    length, result = len(record.moves), record.result
    window.go_to(10)
    pos = window.viewed
    game_move = record.moves[10].move
    legal = sorted(m.uci() for m in pos.legal_moves() if m != game_move)[0]
    assert window.board.interactive
    move(window.board, legal)
    node = window.view_node
    assert node.move.uci() == legal and node.ply == 11 and node.starts_variation()
    assert len(record.moves) == length and record.result == result  # the game stays as it was
    assert window.move_tree.current is node and node in [n for n, _ in window.move_tree.anchors()]
    window.go_to(10)
    window.switch_variation(1)  # no other move was played instead of this one: nothing
    assert window.view_node is record.moves[9]
    move(window.board, game_move.uci())  # the game's move: shown, not added again
    assert window.view_node is record.moves[10] and len(record.moves[9].children) == 2
    window.switch_variation(1)
    assert window.view_node is node
    window.switch_variation(-1)
    assert window.view_node is record.moves[10]


def test_fen_paste_edit_and_settings(window, tmp_path):
    assert not window.load_fen("garbage")
    assert window.load_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert window.game.record.start.fen() == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    window.copy_fen()
    assert QApplication.clipboard().text() == window.viewed.fen()
    for name, cls in (("glyphs", GlyphPieceSet), ("vector", VectorPieceSet), ("cburnett", SvgPieceSet)):
        window.set_piece_set(name)
        assert isinstance(window.board.piece_set, cls)
        assert window.settings.value("piece_set") == name
    window.flip()
    assert window.board.flipped and window.eval_bar.flipped and window.settings.value("flipped") == "true"
    window.toggle_coordinates(False)
    assert not window.board.show_coordinates and window.settings.value("coordinates") == "false"


def test_close_quits_every_engine(window):
    window.start_game(window.profiles[1], None, TimeControl(movetime_ms=60_000))
    assert wait_until(lambda: window.game.thinking_color == WHITE)
    engine = window.game.engines[WHITE]
    window.close()
    assert window.game.engines == [None, None] and engine.state.value == "stopped"


def test_dialog_flows(window, monkeypatch):
    """New game, engines and position editing through their dialogs."""
    from PySide6.QtWidgets import QDialog

    from chessgpu_gui import main_window as mw

    def accept_with(setup):
        def exec_(dialog):
            setup(dialog)
            return QDialog.DialogCode.Accepted
        return exec_

    def new_game_setup(d):
        d.players[WHITE].setCurrentIndex(0)  # Human
        d.players[BLACK].setCurrentText("fake")
        d.mode.setCurrentText("Depth")
        d.depth.setValue(1)

    monkeypatch.setattr(mw.NewGameDialog, "exec", accept_with(new_game_setup))
    window.new_game()
    assert wait_until(lambda: window.game.status == window.game.RUNNING)
    assert window.game.profiles[WHITE] is None and window.game.profiles[BLACK].name == "fake"
    assert window.game.tc == TimeControl(depth=1)
    assert window.settings.value("new_game")["mode"] == "Depth"
    window.game.abort()

    monkeypatch.setattr(mw.ProfilesDialog, "exec", accept_with(lambda d: d.duplicate()))
    window.manage_engines()
    assert [p.name for p in window.profiles] == ["fake", "fake (copy)", "thinker"]
    assert "fake (copy)" in window.settings.value("profiles")
    assert [a.text() for a in window.analysis_group.actions()] == ["fake", "fake (copy)", "thinker"]

    def edit_setup(editor):
        editor.set_position(Position.from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"))

    monkeypatch.setattr(mw.BoardEditor, "exec", accept_with(edit_setup))
    window.edit_position()
    assert window.game.record.start.fen() == "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"


def test_failed_analysis_engine_unchecks_analysis(window):
    window.profiles.append(fake("--crash-on-go", "1", name="crasher"))
    window._fill_analysis_menu()
    window.set_analysis_profile("crasher")
    window.start_analysis()
    assert wait_until(lambda: not window.analyze_action.isChecked())
    assert "crasher" in window.status.text() or "exited" in window.status.text()
    assert not window.outputs["analysis"].isVisible()


# --- Look and feel --------------------------------------------------------------------------------


def sound_window(qapp, tmp_path, *profiles):
    """A window whose sounds go to a list (names, in order)."""
    from chessgpu_gui.sounds import SoundPlayer

    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json(list(profiles) or [fake()]))
    played = []
    player = SoundPlayer(tmp_path / "sounds", backend=lambda path: played.append(path.stem.rsplit("-", 1)[0]))
    w = MainWindow(settings, sounds=player)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    return w, played


def test_theme_and_board_settings(window, qapp):
    from chessgpu_gui.dialogs import ViewSettings
    from chessgpu_gui.theme import BOARD_THEMES, THEMES, current_theme

    try:
        window.set_theme("light")
        assert current_theme() is THEMES["light"] and window.settings.value("theme") == "light"
        assert [a.data() for a in window.theme_group.actions() if a.isChecked()] == ["light"]
        window.set_board_theme("blue")
        assert window.board.board_theme is BOARD_THEMES["blue"] and window.settings.value("board_theme") == "blue"
        v = ViewSettings(theme="light", board_theme="green", piece_set="vector", animation_ms=0, sounds=False,
                         volume=10, show_legal=False, coordinates=False, highlight_last=False, premoves=False,
                         auto_queen=True, figurines=True)
        window.set_view(v)
        b = window.board
        assert (b.animation_ms, b.show_legal, b.show_coordinates, b.highlight_last, b.auto_queen) == \
            (0, False, False, False, True)
        assert b.piece_set.name == "vector" and window.cards[WHITE].material.piece_set.name == "vector"
        assert window.move_tree.figurines and window.outputs["analysis"].figurines
        assert not window.sounds.enabled and window.sounds.volume == 0.1
        assert ViewSettings.load(window.settings) == v
    finally:
        window.set_theme("dark")


def test_cards_follow_the_board(window):
    top = window.board_area.top
    assert top is window.cards[BLACK]
    window.flip()
    assert window.board_area.top is window.cards[WHITE] and window.board_area.bottom is window.cards[BLACK]
    assert window.cards[WHITE].geometry().top() < window.board.geometry().top()
    window.flip()


def test_game_sounds(qapp, tmp_path):
    w, played = sound_window(qapp, tmp_path)
    try:
        w.start_game(None, w.profiles[0], TimeControl(movetime_ms=30))
        assert wait_until(lambda: w.game.status == w.game.RUNNING)
        assert played == ["start"]
        move(w.board, "e2e4")
        assert wait_until(lambda: len(w.game.record.moves) == 2)
        assert played == ["start", "move", "move"]  # e4, then the engine's a5
        w.resign()
        assert played[-1] == "end"
        w.open_pgn_from(str(DATA / "fastchess_sample.pgn"), index=0)
        count = len(played)
        w.go_to(0)
        w.go_to(1)  # one step forward: its move's sound
        w.go_to(5)  # a jump: silent
        assert len(played) == count + 1 and played[-1] in ("move", "capture", "check", "castle", "promote")
    finally:
        w.close()


def test_low_time_warning(qapp, tmp_path):
    w, played = sound_window(qapp, tmp_path)
    try:
        w.start_game(None, w.profiles[0], TimeControl(base_ms=10_500, inc_ms=0))
        assert wait_until(lambda: w.game.status == w.game.RUNNING)
        assert wait_until(lambda: "low_time" in played, 3000)
        assert played.count("low_time") == 1
        assert w.cards[WHITE].clock_state() == "low"
        QTest_wait(300)
        assert played.count("low_time") == 1  # once per game
    finally:
        w.close()


def QTest_wait(ms):
    from PySide6.QtTest import QTest

    QTest.qWait(ms)


def test_premove_plays_when_the_turn_comes(qapp, tmp_path):
    w, _played = sound_window(qapp, tmp_path, fake("--delay", "700", name="slow"))
    try:
        w.start_game(None, w.profiles[0], TimeControl(movetime_ms=2000))
        assert wait_until(lambda: w.game.status == w.game.RUNNING)
        move(w.board, "e2e4")
        assert wait_until(lambda: w.game.thinking_color == BLACK)
        assert w.board.premove_color == WHITE and not w.board.interactive
        move(w.board, "d2d4")  # queued while the engine thinks
        assert w.board.premoves == [Move.from_uci("d2d4")] and len(w.game.record.moves) == 1
        assert wait_until(lambda: len(w.game.record.moves) == 3, 5000)
        assert [m.move.uci() for m in w.game.record.moves] == ["e2e4", "a7a5", "d2d4"]
        assert w.board.premoves == []
    finally:
        w.close()


def test_premove_that_turned_illegal_is_dropped(window):
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    assert wait_until(lambda: window.game.status == window.game.RUNNING)
    window.board.premoves = [Move.from_uci("e2e5")]  # never legal
    window._play_premove()
    assert window.board.premoves == [] and window.game.record.moves == []
    assert "cancelled" in window.status.text()


def test_premoves_off(window):
    from chessgpu_gui.dialogs import ViewSettings

    values = dict(ViewSettings.load(window.settings).__dict__)
    values["premoves"] = False
    window.set_view(ViewSettings(**values))
    window.start_game(None, window.profiles[1], TimeControl(movetime_ms=60_000))  # the thinker waits for stop
    assert wait_until(lambda: window.game.status == window.game.RUNNING)
    move(window.board, "e2e4")
    assert wait_until(lambda: window.game.thinking_color == BLACK)
    assert window.board.premove_color is None
    window.game.abort()


def test_shortcuts_dialog(window):
    from PySide6.QtCore import QTimer

    QTimer.singleShot(100, lambda: window.shortcuts_dialog.reject())
    window.show_shortcuts()
    from PySide6.QtWidgets import QTableWidget

    table = window.shortcuts_dialog.findChild(QTableWidget)
    names = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "New game" in names and "Settings" in names and "Flip board" in names


def right_drag(board, a, b):
    for kind, buttons, name in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.RightButton, a),
                                (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton, b)):
        p = board.square_rect(parse_square(name)).center()
        event = QMouseEvent(kind, QPointF(p), QPointF(board.mapToGlobal(QPoint(int(p.x()), int(p.y())))),
                            Qt.MouseButton.RightButton, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(board, event)


def test_variations_comments_and_shapes(window, tmp_path):
    assert window.open_pgn_from(str(DATA / "annotated.pgn"))
    record = window.game.record
    nc3 = record.moves[12]
    bd2 = nc3.parent.children[1]  # 7. Bd2, instead of 7. Nc3
    # A variation's move: shown; Back and Forward then follow its line.
    click_move(window.move_tree, bd2)
    assert window.view_node is bd2 and window.board.position == bd2.position
    window.go_to(window.view_ply + 1)
    assert window.view_node is bd2.children[0]
    window.go_to(window.view_ply - 2)
    assert window.view_node is record.moves[11]
    window.go_to_node(bd2.children[0])
    window.go_to_end()
    assert window.view_node is bd2.end() and window.view_node.san == "d5"
    # Arrows and circles belong to positions: from the PGN, and drawn.
    d5 = record.moves[16]
    window.go_to_node(d5)
    assert d5.arrows and window.board.shape_arrows == d5.arrows
    window.go_to_node(record.moves[0])
    assert window.board.shape_arrows == {}
    right_drag(window.board, "g1", "f3")
    assert record.moves[0].arrows == {(parse_square("g1"), parse_square("f3")): "green"}
    window.go_to_node(d5)
    window.go_to_node(record.moves[0])
    assert window.board.shape_arrows == record.moves[0].arrows
    # Comments: the comment box edits the shown move's; the tree shows it soon after.
    window.go_to_node(record.moves[1])
    assert window.comment_box.node is record.moves[1]
    window.comment_box.setPlainText("Solid")
    assert record.moves[1].comment == "Solid"
    assert wait_until(lambda: "Solid" in window.move_tree.toPlainText())
    window.go_to_node(record.root)
    assert window.comment_box.toPlainText() == record.root.comment
    # NAGs from the context menu.
    window._tree_action("nag:1", record.moves[1])
    assert record.moves[1].nags == [1] and "e5!" in [t for _, t in window.move_tree.anchors()]
    window._tree_action("nag:1", record.moves[1])
    assert record.moves[1].nags == []
    window._tree_action("nag:3", record.moves[1])
    # Promoting, making main, deleting.
    window._tree_action("make_main", bd2)
    assert record.moves[12] is bd2 and record.moves[-1].san == "d5"
    window._tree_action("promote", nc3)
    assert record.moves[12] is nc3
    window.go_to_node(bd2.end())
    window._tree_action("delete", bd2)
    assert window.view_node is record.moves[11] and not record.contains(bd2)
    assert len(record.moves[11].children) == 1
    window._tree_action("copy_line", record.moves[3])
    assert QApplication.clipboard().text() == "1. e4 e5 2. Nf3 Nc6"
    # Saving keeps it all.
    path = tmp_path / "annotated.pgn"
    window.save_pgn_to(str(path))
    assert window.open_pgn_from(str(path))
    again = window.game.record
    assert again is not record and [n.san for n in again.moves] == [n.san for n in record.moves]
    assert again.moves[1].comment == "Solid" and again.moves[1].nags == [3]
    assert again.moves[0].arrows == record.moves[0].arrows and again.moves[16].arrows == d5.arrows
    assert len(again.nodes()) == len(record.nodes()) and again.root.comment == record.root.comment


def test_tree_edits_wait_for_the_game(window):
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    assert wait_until(lambda: window.game.status == window.game.RUNNING)
    move(window.board, "e2e4")
    assert wait_until(lambda: len(window.game.record.moves) == 2)
    e4 = window.game.record.moves[0]
    assert not window.move_tree.can_edit
    window._tree_action("delete", e4)
    assert window.game.record.contains(e4) and "Finish or abort" in window.status.text()
    window._tree_action("nag:1", e4)  # annotating is fine
    assert e4.nags == [1]
    window.game.abort()
    assert window.move_tree.can_edit


def analysis_window(qapp, tmp_path, profiles, **values):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json(profiles))
    for k, v in values.items():
        settings.setValue(k, v)
    w = MainWindow(settings)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    return w


def test_multipv_analysis(qapp, tmp_path):
    w = analysis_window(qapp, tmp_path, [fake("--multipv", name="multi"), fake(name="single")], analysis_lines=3)
    try:
        assert w.lines_box.value() == 3
        w.start_analysis()
        out = w.outputs["analysis"]
        assert wait_until(lambda: len(out.lines) == 3)
        best = sorted(m.uci() for m in Position.start().legal_moves())[:3]
        assert [out.lines[k][1][0] for k in (1, 2, 3)] == best
        assert [a.uci() for a in w.board.arrows] == best and [c[3] for c in w.board.arrow_colors] == [170, 110, 80]
        # Fewer lines: the engine is told, the analysis starts again.
        w.lines_box.setValue(2)
        assert wait_until(lambda: sorted(out.lines) == [1, 2] and len(w.board.arrows) == 2)
        assert str(w.settings.value("analysis_lines")) == "2"
        assert "setoption name MultiPV value 2" in w.log_view.toPlainText()
        # A line's first move: played as a variation, shown.
        line = list(out.lines[2][1])
        out.pv_clicked.emit(2, 1)
        node = w.view_node
        assert node.uci_moves() == line[:1] and w.game.record.contains(node)
        assert wait_until(lambda: out.root == node.position and len(out.lines) == 2)
        # Its second move: both moves.
        w.go_to(0)
        assert wait_until(lambda: out.root == w.viewed and len(out.lines) == 2)
        out.pv_clicked.emit(2, 2)
        node = w.view_node
        assert node.uci_moves() == line[:2] and node.parent.uci_moves() == line[:1]
        assert wait_until(lambda: out.root == node.position and len(out.lines) == 2)
        # An engine without MultiPV: one line, and a word about it.
        w.set_analysis_profile("single")
        assert wait_until(lambda: w.analysis.engine.running and len(out.lines) == 1)
        w.set_analysis_lines(3)
        assert "no MultiPV" in w.status.text()
        assert wait_until(lambda: len(out.lines) == 1)
    finally:
        w.close()


def test_threat_mode(window):
    window.toggle_threat(True)
    assert window.threat_mode and window.analyze_action.isChecked()
    assert wait_until(lambda: window.board.arrows)
    assert window.board.arrows[0].uci() == "a7a5"  # Black's first move in UCI order: the fake engine's choice
    assert window.board.arrow_colors[0][:3] == THREAT_ARROW[:3] and "Threat" in window.outputs["analysis"].title.text()
    assert window.eval_bar.score is None  # a threat's score isn't the position's
    window.outputs["analysis"].pv_clicked.emit(1, 1)  # a threat line isn't playable here
    assert window.view_node is window.game.record.root
    assert window.load_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")  # in check: no threat to show
    assert "none to show" in window.outputs["analysis"].title.text() and not window.analysis.running
    window.toggle_threat(False)
    assert wait_until(lambda: window.board.arrows and window.board.arrow_colors[0][:3] == ENGINE_ARROW[:3])
    window.stop_analysis()
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    window.toggle_threat(True)  # not during a game
    assert not window.threat_mode and not window.threat_action.isChecked()
    window.game.abort()


def test_hint(window):
    window.start_game(None, window.profiles[0], TimeControl(movetime_ms=30))
    assert wait_until(lambda: window.game.status == window.game.RUNNING and window.game.human_to_move)
    assert window.hint_action.isEnabled()
    window.hint()
    assert wait_until(lambda: window.board.arrows, 5000)
    first = sorted(m.uci() for m in Position.start().legal_moves())[0]
    assert window.board.arrows[0].uci() == first and window.status.text() == "Hint: a3 (+0.15)"
    move(window.board, "e2e4")
    assert wait_until(lambda: len(window.game.record.moves) == 2)
    assert window.board.arrows == []
    window._hint_for = window.game.record.root  # a hint arriving after the game moved on: not shown
    window._hint_ready(first, None)
    assert window.board.arrows == [] and window._hint_for is None
    window.game.abort()
    assert not window.hint_action.isEnabled()
    window.hint()
    assert "Hints are for your move" in window.status.text()
