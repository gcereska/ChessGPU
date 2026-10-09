"""Widgets and dialogs: piece sets, the board (geometry and input), panels,
the options form, the new-game dialog and the board editor."""

import pathlib
import sys

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit, QSpinBox

from chessgpu_gui.board_view import BoardView
from chessgpu_gui.dialogs import BoardEditor, NewGameDialog
from chessgpu_gui.game import GameController, GameRecord, TimeControl
from chessgpu_gui.options_dialog import OptionsForm, ProfileDialog, ProfilesDialog
from chessgpu_gui.move_tree import MoveTreeView
from chessgpu_gui.panels import (EngineOutput, PlayerCard, EvalBar, EvalGraph, format_clock, pv_san, white_score,
                                 win_probability)
from chessgpu_gui.pieces import PIECE_SETS, PIECES, GlyphPieceSet, VectorPieceSet, make_piece_set
from chessgpu_gui.profiles import EngineProfile
from chessgpu_gui.rules import BLACK, START_FEN, WHITE, Move, Position, parse_square
from chessgpu_gui.score import MoveEval, Score
from chessgpu_gui.uci import Info, UciOption
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))


# --- Piece sets ----------------------------------------------------------------------------


def ink(image: QImage) -> tuple[int, float]:
    """(pixels with any opacity, their mean lightness)."""
    count, light = 0, 0.0
    for y in range(image.height()):
        for x in range(image.width()):
            c = QColor(image.pixelColor(x, y))
            if c.alpha() > 40:
                count += 1
                light += c.lightnessF()
    return count, (light / count if count else 0.0)


@pytest.mark.parametrize("name", list(PIECE_SETS) + ["glyphs-fallback"])
def test_piece_sets_draw_every_piece(qapp, name):
    ps = GlyphPieceSet(use_font=False) if name == "glyphs-fallback" else make_piece_set(name)
    for piece in PIECES:
        image = ps.pixmap(piece, 48).toImage()
        assert image.width() == 48
        count, _ = ink(image)
        assert count > 48 * 48 * 0.12, f"{name} {piece} draws almost nothing"
    white = ink(ps.pixmap("Q", 48).toImage())[1]
    black = ink(ps.pixmap("q", 48).toImage())[1]
    assert white > black + 0.15, f"{name}: white and black queens look alike"
    assert ps.pixmap("K", 48) is ps.pixmap("K", 48)  # cached


def test_piece_set_registry(qapp):
    assert set(PIECE_SETS) == {"cburnett", "glyphs", "vector"}
    assert isinstance(make_piece_set("no such set"), type(make_piece_set("cburnett")))
    assert make_piece_set("vector").name == "vector"


# --- Board -----------------------------------------------------------------------------------


def mouse(widget, kind, pos, button=Qt.MouseButton.LeftButton, buttons=None):
    buttons = button if buttons is None else buttons
    event = QMouseEvent(kind, QPointF(pos), QPointF(widget.mapToGlobal(QPoint(int(pos.x()), int(pos.y())))),
                        button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def center(board, name):
    return board.square_rect(parse_square(name)).center()


def click(board, name, button=Qt.MouseButton.LeftButton):
    p = center(board, name)
    mouse(board, QEvent.Type.MouseButtonPress, p, button)
    mouse(board, QEvent.Type.MouseButtonRelease, p, button, Qt.MouseButton.NoButton)


@pytest.fixture
def board(qapp):
    b = BoardView(VectorPieceSet())
    b.resize(400, 400)
    b.moves = Recorder(b.move_made)
    return b


@pytest.mark.parametrize("flipped", [False, True])
def test_board_geometry(board, flipped):
    board.set_flipped(flipped)
    for sq in range(64):
        assert board.square_at(board.square_rect(sq).center()) == sq
    a1 = board.square_rect(parse_square("a1"))
    assert (a1.left() < 200 and a1.top() > 200) != flipped  # a1 at the bottom left unless flipped
    assert board.square_at(QPointF(-5, 10)) is None
    board.align_left = True
    board.resize(500, 400)
    assert board.board_rect().left() == 0 and board.board_rect().width() == 400


def test_click_click_and_drag_moves(board):
    board.set_interactive(True)
    click(board, "e2")
    click(board, "e4")
    assert board.moves.calls == [(Move.from_uci("e2e4"),)]
    mouse(board, QEvent.Type.MouseButtonPress, center(board, "g1"))
    mouse(board, QEvent.Type.MouseMove, center(board, "g2"), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    mouse(board, QEvent.Type.MouseMove, center(board, "f3"), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    mouse(board, QEvent.Type.MouseButtonRelease, center(board, "f3"), Qt.MouseButton.LeftButton,
          Qt.MouseButton.NoButton)
    assert board.moves.calls[-1] == (Move.from_uci("g1f3"),)


def test_board_rejects_bad_input(board):
    board.set_interactive(True)
    click(board, "e2")
    click(board, "e5")  # not a target: no move
    click(board, "e7")  # the opponent's piece: no selection
    click(board, "e5")
    click(board, "a3")  # nothing selected
    click(board, "e2")
    click(board, "e4", Qt.MouseButton.RightButton)  # right click cancels
    click(board, "e4")
    assert board.moves.calls == []
    board.set_interactive(False)
    click(board, "e2")
    click(board, "e4")
    assert board.moves.calls == []


def test_promotion_choice(board):
    board.set_position(Position.from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1"))
    board.set_interactive(True)
    asked = []
    board.promotion_chooser = lambda color, sq: asked.append((color, sq)) or "n"
    click(board, "b7")
    click(board, "b8")
    assert board.moves.calls == [(Move.from_uci("b7b8n"),)]
    assert asked == [(WHITE, parse_square("b8"))]
    board.promotion_chooser = lambda color, sq: None  # cancelled
    click(board, "b7")
    click(board, "b8")
    assert len(board.moves.calls) == 1


def test_flipped_board_input(board):
    board.set_flipped(True)
    board.set_position(Position.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))
    board.set_interactive(True)
    click(board, "d7")
    click(board, "d5")
    assert board.moves.calls == [(Move.from_uci("d7d5"),)]
    assert center(board, "h1").x() < 100  # h1 at the top left when flipped
    assert center(board, "h1").y() < 100


def test_edit_mode_reports_squares(board):
    board.edit_mode = True
    clicks = Recorder(board.square_clicked)
    click(board, "c3")
    click(board, "h8", Qt.MouseButton.RightButton)
    assert [(c[0], c[1]) for c in clicks.calls] == [(parse_square("c3"), Qt.MouseButton.LeftButton),
                                                     (parse_square("h8"), Qt.MouseButton.RightButton)]
    assert board.moves.calls == []


def test_board_paints_every_state(board):
    board.set_position(Position.from_fen("4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1"), Move.from_uci("a2a1"))
    board.set_arrows([Move.from_uci("e1g1")])
    board.set_interactive(True)
    image = board.grab().toImage()  # last move, coordinates, arrows; black to move, not in check
    assert image.width() == 400
    board.set_position(Position.from_fen("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1"))  # check highlight
    click(board, "e8")  # selection with targets
    assert board.grab().toImage().width() == 400


# --- Panels --------------------------------------------------------------------------------------


def test_format_clock():
    assert format_clock(0) == "0.0" and format_clock(-50) == "0.0"
    assert format_clock(9_950) == "9.9" and format_clock(10_000) == "0:10"
    assert format_clock(65_400) == "1:05" and format_clock(3_600_000) == "1:00:00"
    assert format_clock(3_725_000) == "1:02:05"


def test_scores_for_display():
    assert win_probability(Score(cp=0)) == 0.5 and win_probability(None) == 0.5
    assert win_probability(Score(mate=3)) == 1.0 and win_probability(Score(mate=-3)) == 0.0
    assert 0.5 < win_probability(Score(cp=100)) < win_probability(Score(cp=300)) < 1
    assert white_score(Score(cp=30), BLACK) == Score(cp=-30) and white_score(Score(cp=30), WHITE) == Score(cp=30)
    assert white_score(None, WHITE) is None


def test_pv_san():
    start = Position.start()
    assert pv_san(start, ["e2e4", "e7e5", "g1f3"]) == "1. e4 e5 2. Nf3"
    black = start.push(Move.from_uci("e2e4"))
    assert pv_san(black, ["e7e5", "g1f3"]) == "1... e5 2. Nf3"
    assert pv_san(start, ["e2e4", "e2e4", "g1f3"]) == "1. e4"  # stops at the illegal move
    assert pv_san(start, []) == ""


def record_of(*sans, fen=START_FEN):
    r = GameRecord(Position.from_fen(fen))
    for s in sans:
        r.push(r.position.parse_san(s))
    return r


def test_move_list(qapp):
    view = MoveTreeView()
    view.resize(300, 300)
    selected = Recorder(view.node_selected)
    record = record_of("e4", "e5", "Nf3")
    record.moves[1].eval = MoveEval(Score(cp=20), 8, 0.5)
    view.set_tree(record, record.moves[2])
    assert [text for _, text in view.anchors()] == ["e4", "e5", "Nf3"]
    assert [node for node, _ in view.anchors()] == record.moves
    assert MoveTreeView.tooltip(record.moves[1]) == "+0.20/8 0.500s"
    assert view.toPlainText().split() == ["1.", "e4", "e5", "2.", "Nf3"]
    view.anchorClicked.emit(QUrl("n:1"))
    assert selected.calls == [(record.moves[1],)]
    view.anchorClicked.emit(QUrl("n:99"))  # no such move: nothing
    view.anchorClicked.emit(QUrl("https://example.com"))
    assert len(selected.calls) == 1
    black_first = record_of("e5", "Nf3", fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    view.set_tree(black_first, black_first.root)
    assert view.toPlainText().split() == ["1.", "…", "e5", "2.", "Nf3"]


def test_engine_output(qapp):
    out = EngineOutput("Engine")
    root = Position.start().push(Move.from_uci("e2e4"))  # black to move
    out.show_info(Info(depth=5, seldepth=9, score=Score(cp=40), nodes=1_500_000, nps=3_000_000, time_ms=500,
                       pv=["e7e5", "g1f3"]), root)
    assert out.fields["Score"].text() == "-0.40" and out.score == Score(cp=-40)  # White's view
    assert out.fields["Depth"].text() == "5/9" and out.fields["Nodes"].text() == "1.5M"
    assert out.fields["NPS"].text() == "3.0M" and out.fields["Time"].text() == "0.5 s"
    assert out.line_text(1) == "1... e5 2. Nf3" and out.best == "e7e5"
    out.show_info(Info(depth=6, score=Score(cp=90), bound="lower"), root)  # a bound doesn't replace the score
    assert out.fields["Score"].text() == "-0.40" and out.fields["Depth"].text() == "6"
    out.clear("Other")
    assert out.title.text() == "Other" and out.score is None and out.line_text(1) == "" and out.lines == {}


def test_engine_output_lines(qapp):
    out = EngineOutput("Analysis")
    clicks = Recorder(out.pv_clicked)
    root = Position.start()
    out.set_max_lines(2)
    out.show_info(Info(depth=8, multipv=1, score=Score(cp=30), pv=["e2e4", "e7e5", "g1f3"]), root)
    out.show_info(Info(depth=8, multipv=2, score=Score(cp=20), pv=["d2d4", "d7d5"]), root)
    out.show_info(Info(depth=8, multipv=3, score=Score(cp=10), pv=["c2c4"]), root)  # beyond the two wanted
    assert sorted(out.lines) == [1, 2] and out.best == "e2e4" and out.score == Score(cp=30)
    assert out.line_text(1) == "1. e4 e5 2. Nf3" and out.line_text(2) == "1. d4 d5"
    shown = [label for label in out.line_labels if not label.isHidden()]
    assert len(shown) == 2 and "+0.30" in shown[0].text() and "+0.20" in shown[1].text()
    out.show_info(Info(depth=9, multipv=2, score=Score(cp=35), pv=["d2d4", "g8f6"]), root)  # line 2 at depth 9
    assert out.score == Score(cp=30) and out.lines[2][0] == Score(cp=35) and out.fields["Score"].text() == "+0.30"
    out.show_info(Info(depth=9, multipv=2, score=Score(cp=50), bound="lower", pv=["d2d4"]), root)
    assert out.lines[2] == (Score(cp=35), ["d2d4"])  # a bound keeps the line's last exact score
    shown[1].linkActivated.emit("2:1")
    shown[0].linkActivated.emit("1:3")
    shown[0].linkActivated.emit("junk")
    assert clicks.calls == [(2, 1), (1, 3)]
    other = root.push(Move.from_uci("e2e4"))
    out.show_info(Info(depth=1, multipv=1, score=Score(cp=-30), pv=["c7c5"]), other)  # a new position: new lines
    assert out.lines == {1: (Score(cp=30), ["c7c5"])} and out.line_text(1) == "1... c5"
    out.set_max_lines(1)
    out.show_info(Info(depth=2, multipv=2, score=Score(cp=-10), pv=["e7e5"]), other)  # beyond the one wanted now
    assert sorted(out.lines) == [1] and sum(not label.isHidden() for label in out.line_labels) == 1
    out.figurines = True
    out.show_info(Info(depth=2, multipv=1, score=Score(cp=-30), pv=["b8c6"]), other)
    assert out.line_text(1) == "1... ♘c6"


def test_eval_bar_and_graph(qapp):
    bar = EvalBar()
    bar.resize(26, 300)
    bar.set_score(Score(cp=200))
    assert bar.white_fraction() > 0.7
    bar.set_flipped(True)
    assert bar.grab().toImage().height() == 300
    graph = EvalGraph()
    graph.resize(400, 80)
    plies = Recorder(graph.ply_selected)
    record = record_of("e4", "e5", "Nf3", "Nc6")
    record.moves[0].eval = MoveEval(Score(cp=30), 10)
    record.moves[1].eval = MoveEval(Score(cp=10), 10)  # black's view: White -0.10
    record.moves[3].eval = MoveEval(Score(mate=2), 10)  # black mates: White's chances 0
    graph.set_record(record, 2)
    assert graph.points[0] > 0.5 and graph.points[1] < 0.5 and graph.points[2] is None and graph.points[3] == 0.0
    for ply in range(5):
        assert graph.ply_at(graph.x_of(ply)) == ply
    assert graph.grab().toImage().width() == 400
    mouse(graph, QEvent.Type.MouseButtonPress, QPointF(graph.x_of(3), 40))
    assert plies.calls == [(3,)]


def test_player_card(qapp):
    g = GameController()
    g.new_game(None, None, TimeControl(base_ms=65_000))
    white, black = PlayerCard(g, VectorPieceSet(), WHITE), PlayerCard(g, VectorPieceSet(), BLACK)
    for card in (white, black):
        card.refresh()
    assert white.clock.text() in ("1:05", "1:04") and black.clock.text() == "1:05"
    assert white.name.text() == "Human" and white.sub.text() == "to move" and black.sub.text() == "Black"
    assert white.clock_state() == "active" and black.clock_state() == ""
    assert white.property("active") == "true" and black.property("active") == "false"
    g.abort()
    white.refresh()
    assert white.clock_state() == "" and white.property("active") == "false"
    untimed = GameController()
    untimed.new_game(None, None, TimeControl(depth=3))
    card = PlayerCard(untimed, VectorPieceSet(), WHITE)
    assert card.clock.isHidden()
    untimed.abort()


# --- Options form and profiles ---------------------------------------------------------------------


OPTIONS = [UciOption("Hash", "spin", "16", 1, 1024), UciOption("Style", "combo", "Normal", vars=["Normal", "Wild"]),
           UciOption("Ponder", "check", "false"), UciOption("Book", "string", ""),
           UciOption("Clear Hash", "button"), UciOption("Huge", "spin", "5", 0, 2**40)]


def test_options_form(qapp):
    form = OptionsForm(OPTIONS, {"hash": "64", "Style": "wild", "Gone": "1", "Ponder": "bogus"})
    assert set(form.inputs) == {"Hash", "Style", "Ponder", "Book", "Huge"}  # no buttons
    assert isinstance(form.inputs["Hash"], QSpinBox) and form.inputs["Hash"].value() == 64
    assert isinstance(form.inputs["Style"], QComboBox) and form.inputs["Style"].currentText() == "Wild"
    assert isinstance(form.inputs["Ponder"], QCheckBox) and not form.inputs["Ponder"].isChecked()
    assert isinstance(form.inputs["Huge"], QLineEdit)  # too wide for a spin box
    assert form.values() == {"Hash": "64", "Style": "Wild", "Gone": "1"}
    form.inputs["Hash"].setValue(16)
    form.inputs["Book"].setText("x.bin")
    form.inputs["Huge"].setText("not a number")
    assert form.values() == {"Style": "Wild", "Book": "x.bin", "Gone": "1"}


def test_profile_dialog_reads_engine_options(qapp):
    profile = EngineProfile("fake", sys.executable, [FAKE], {"Hash": "32"})
    dialog = ProfileDialog(profile)
    assert wait_until(lambda: dialog.form is not None)
    assert "Hash" in dialog.form.inputs and dialog.form.inputs["Hash"].value() == 32
    dialog.form.inputs["Style"].setCurrentText("Wild")
    result = dialog.result_profile()
    assert result.options == {"Hash": "32", "Style": "Wild"} and result.args == [FAKE]
    dialog.done(0)
    broken = ProfileDialog(EngineProfile("missing", str(pathlib.Path(FAKE).with_name("missing.exe"))))
    assert wait_until(lambda: "Couldn't read options" in broken.status.text())
    assert broken.result_profile().options == {}
    broken.done(0)


def test_profiles_dialog_list_operations(qapp):
    profiles = [EngineProfile("a", "a.exe"), EngineProfile("b", "b.exe", options={"Eval": "material"})]
    dialog = ProfilesDialog(profiles)
    assert dialog.list.count() == 2 and "Eval=material" in dialog.list.item(1).text()
    dialog.list.setCurrentRow(1)
    dialog.duplicate()
    assert [p.name for p in dialog.profiles] == ["a", "b", "b (copy)"]
    dialog.list.setCurrentRow(0)
    dialog.remove()
    assert [p.name for p in dialog.profiles] == ["b", "b (copy)"]
    assert [p.name for p in profiles] == ["a", "b"]  # the caller's list is untouched until OK


# --- New game dialog and board editor ------------------------------------------------------------


def test_new_game_dialog(qapp):
    profiles = [EngineProfile("bot", "bot.exe")]
    current = Position.start().push(Move.from_uci("d2d4"))
    d = NewGameDialog(profiles, current)
    assert d.player(WHITE) is None and d.player(BLACK) is profiles[0]  # defaults: human against the first engine
    d.minutes.setValue(3)
    d.increment.setValue(2)
    assert d.time_control() == TimeControl(base_ms=180_000, inc_ms=2_000)
    d.mode.setCurrentText("Time per move")
    d.movetime.setValue(0.5)
    assert d.time_control() == TimeControl(movetime_ms=500)
    d.mode.setCurrentText("Depth")
    assert d.time_control() == TimeControl(depth=8)
    d.mode.setCurrentText("Nodes")
    assert d.time_control().nodes == 1_000_000
    d.start_current.setChecked(True)
    assert d.start_position() == current
    d.start_fen.setChecked(True)
    d.fen.setText("not a fen")
    d.accept()
    assert d.result() == 0 and "Invalid FEN" in d.error.text()
    d.fen.setText("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate: nothing to play
    d.accept()
    assert "already over" in d.error.text()
    settings = d.settings()
    again = NewGameDialog(profiles, current, settings)
    assert again.mode.currentText() == "Nodes" and again.players[BLACK].currentText() == "bot"


def test_board_editor(qapp):
    editor = BoardEditor(Position.start(), VectorPieceSet())
    ok = editor.buttons.button(editor.buttons.StandardButton.Ok)
    assert ok.isEnabled()
    editor.clear()
    assert not ok.isEnabled() and "king" in editor.problem.text()
    editor._set_brush("K")
    editor._clicked(parse_square("e1"), Qt.MouseButton.LeftButton)
    editor._set_brush("k")
    editor._clicked(parse_square("e8"), Qt.MouseButton.LeftButton)
    editor._set_brush("p")
    editor._clicked(parse_square("d5"), Qt.MouseButton.LeftButton)
    editor._set_brush("P")
    editor._clicked(parse_square("e5"), Qt.MouseButton.LeftButton)
    assert ok.isEnabled()
    assert not any(box.isEnabled() or box.isChecked() for box in editor.castle.values())  # no rooks at home
    assert [editor.ep.itemText(i) for i in range(editor.ep.count())] == ["–", "d6"]
    editor.ep.setCurrentIndex(1)
    assert editor.result_position().fen() == "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    editor._set_brush("R")
    editor._clicked(parse_square("h1"), Qt.MouseButton.LeftButton)
    assert editor.castle["K"].isEnabled() and not editor.castle["Q"].isEnabled()
    editor.castle["K"].setChecked(True)
    assert editor.result_position().fen() == "4k3/8/8/3pP3/8/8/8/4K2R w K d6 0 1"
    editor._clicked(parse_square("h1"), Qt.MouseButton.LeftButton)  # same piece again: removed
    assert not editor.castle["K"].isChecked() and ok.isEnabled()
    editor.side.setCurrentIndex(1)  # black to move: no white double push to capture
    assert editor.ep.count() == 1 and editor.draft().ep is None
    editor._clicked(parse_square("e5"), Qt.MouseButton.RightButton)  # right click erases
    assert editor.cells[parse_square("e5")] == "."
    editor.fen.setText("8/8/8/8/8/8/8/8 w - - 0 1")  # loads even though invalid, and says why
    editor._fen_edited()
    assert "king" in editor.problem.text() and not ok.isEnabled()
    editor.initial()
    assert editor.result_position().fen() == START_FEN
