"""Game review: judging moves from the positions' evaluations, the summary
per side, annotating, the review controller with the scripted engine (and
the real one, if built), and the review in the main window."""

import math
import os
import pathlib
import sys

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from chessgpu_gui.game import GameRecord
from chessgpu_gui.main_window import MainWindow
from chessgpu_gui.pgn import read_pgn, write_pgn
from chessgpu_gui.profiles import EngineProfile, profiles_to_json
from chessgpu_gui.review import (BLUNDER, INACCURACY, MISTAKE, ReviewController, accuracy_of, annotate,
                                 expected_score, judge, kind_of, summarize)
from chessgpu_gui.review_panel import REVIEW_SPEEDS, ReviewDialog, ReviewPanel
from chessgpu_gui.rules import BLACK, START_FEN, WHITE, Move, Position
from chessgpu_gui.score import PositionEval, Score
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))
GAME = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"]


def fake(*flags, name="fake"):
    return EngineProfile(name, sys.executable, [FAKE, *flags])


def record_of(sans, fen=START_FEN):
    r = GameRecord(Position.from_fen(fen))
    for s in sans:
        r.push(r.position.parse_san(s))
    return r


def reviewed(sans, scores, bests=None):
    """A record whose positions (start, then after each move) carry White's-view
    centipawn scores (None: unreviewed), and the engine's best moves (UCI)."""
    r = record_of(sans)
    bests = bests or {}
    for i, node in enumerate([r.root] + r.moves):
        if scores[i] is not None:
            score = scores[i] if isinstance(scores[i], Score) else Score(cp=scores[i])
            node.review = PositionEval(score, 10, tuple(bests.get(i, ("h2h3",) if node.position.turn == WHITE
                                                                    else ("h7h6",))))
    return r


# --- Judging ---------------------------------------------------------------------------------------


def test_thresholds_and_accuracy():
    assert kind_of(INACCURACY - 1e-9) is None and kind_of(INACCURACY) == "inaccuracy"
    assert kind_of(MISTAKE) == "mistake" and kind_of(BLUNDER - 1e-9) == "mistake" and kind_of(BLUNDER) == "blunder"
    assert accuracy_of(0) == 100 and accuracy_of(0.25) == pytest.approx(100 * math.exp(-1))
    assert expected_score(Score(cp=0), WHITE) == 0.5 and expected_score(Score(cp=400), BLACK) == pytest.approx(1 / 11)
    assert expected_score(Score(mate=2), WHITE) == 1.0 and expected_score(Score(mate=2), BLACK) == 0.0


def test_judge_and_summarize():
    # White: e4 fine, Nf3 blunders (+50 -> -200 for White); Black: Nc6 too (-200 -> +120).
    r = reviewed(["e4", "e5", "Nf3", "Nc6", "Bc4"], [20, 25, 50, -200, 120, 110])
    js = judge(r)
    assert [j.node.san for j in js] == ["e4", "e5", "Nf3", "Nc6", "Bc4"]
    by = {j.node.san: j for j in js}
    assert by["e4"].kind is None and by["e4"].mover == WHITE
    nf3 = by["Nf3"]
    assert nf3.kind == "blunder" and nf3.cp_loss == 250 and nf3.best == "h2h3" and nf3.best_line == ("h2h3",)
    assert nf3.loss == pytest.approx(expected_score(Score(cp=50), WHITE) - expected_score(Score(cp=-200), WHITE))
    nc6 = by["Nc6"]
    assert nc6.mover == BLACK and nc6.kind == "blunder" and nc6.cp_loss == 320
    assert by["Bc4"].kind is None and by["Bc4"].cp_loss == 10
    sides = summarize(js)
    assert sides[WHITE].moves == 3 and sides[WHITE].blunders == 1 and sides[BLACK].blunders == 1
    assert sides[WHITE].acpl == pytest.approx((0 + 250 + 10) / 3)
    assert sides[WHITE].accuracy == pytest.approx(sum(j.accuracy for j in js if j.mover == WHITE) / 3)
    # The engine's own move is never a mistake, whatever the scores say (depth noise).
    r = reviewed(["e4", "e5"], [20, -300, -300], bests={0: ("e2e4",)})
    assert judge(r)[0].kind is None and judge(r)[0].loss == 0 and judge(r)[0].best is None
    # Unreviewed positions: no verdict on the moves around them.
    r = reviewed(["e4", "e5", "Nf3"], [20, None, 25, 30])
    assert [j.node.san for j in judge(r)] == ["Nf3"]
    # A mate let slip but still winning: a mistake, not a blunder.
    r = reviewed(["e4", "e5"], [Score(mate=3), 200, 210])
    assert judge(r)[0].kind == "mistake"
    assert summarize([])[WHITE].accuracy == 0 and summarize([])[BLACK].moves == 0


def test_annotate():
    r = reviewed(["e4", "e5", "Nf3", "Nc6"], [20, 25, 50, -200, -190], bests={2: ("f1c4", "g8f6", "d2d3")})
    nf3 = r.moves[2]
    nf3.nags = [1, 16]
    nf3.comment = "Natural"
    js = judge(r)
    assert annotate(r, js) == 1
    assert nf3.nags == [16, 4] and nf3.comment == "Blunder. Bc4 was best. Natural"
    variation = r.moves[1].children[1]
    assert variation.san == "Bc4" and [n.san for n in variation.line()[3:]] == ["Bc4", "Nf6", "d3"]
    nodes = len(r.nodes())
    assert annotate(r, judge(r)) == 1  # again: nothing new
    assert nf3.comment == "Blunder. Bc4 was best. Natural" and len(r.nodes()) == nodes
    r2 = reviewed(["e4", "e5", "Nf3"], [20, 25, 50, -200], bests={2: ("f1c4",)})
    annotate(r2, judge(r2), variations=False)
    assert len(r2.moves[1].children) == 1 and r2.moves[2].nags == [4]


def test_review_round_trips_through_pgn():
    r = reviewed(["e4", "e5", "Nf3"], [20, 25, 50, -200])
    back = GameRecord.from_pgn(read_pgn(write_pgn(r.to_pgn()))[0])
    assert [n.review.score for n in [back.root] + back.moves] == [Score(cp=c) for c in (20, 25, 50, -200)]
    assert [j.kind for j in judge(back)] == [j.kind for j in judge(r)]


# --- The controller ------------------------------------------------------------------------------------


def run(controller, record, profile, **kw):
    done = Recorder(controller.finished)
    controller.start(record, profile, **kw)
    assert wait_until(lambda: done.calls, 20_000)
    return done.calls[0][0]


def test_review_controller(qapp):
    c = ReviewController()
    progress = Recorder(c.progress)
    r = record_of(GAME)
    assert run(c, r, fake("--blunder-at", "5"), movetime_ms=20) is True
    assert all(n.review is not None for n in [r.root] + r.moves) and not c.running and c.engine is None
    assert progress.calls[0] == (0, 11) and progress.calls[-1] == (11, 11)
    assert r.root.review.score == Score(cp=15) and r.moves[0].review.score == Score(cp=-15)
    assert r.moves[4].review.score == Score(cp=-500) and r.moves[4].review.pv
    kinds = [(j.node.ply, j.kind) for j in judge(r) if j.kind]
    assert kinds == [(5, "blunder")]  # 3. Bc4, the 5th move, by White
    # A game ending in mate: the last position needs no engine.
    mate = record_of(["f3", "e5", "g4", "Qh4#"])
    assert run(c, mate, fake(), depth=3) is True
    assert mate.moves[-1].review == PositionEval(Score(mate=0), 0)  # White is mated (White's view)
    scholar = record_of(["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"])
    assert run(c, scholar, fake(), depth=3) is True
    assert scholar.moves[-1].review == PositionEval(Score(mate=0, given=True), 0)  # Black is
    assert judge(scholar)[-1].kind is None  # mating isn't a mistake


def test_review_cancel_and_failure(qapp):
    c = ReviewController()
    r = record_of(GAME)
    c.progress.connect(lambda done, _total: c.cancel() if done == 3 else None)
    assert run(c, r, fake("--delay", "30"), movetime_ms=20) is False
    reviewed_now = [n.review is not None for n in [r.root] + r.moves]
    assert reviewed_now[:3] == [True] * 3 and not any(reviewed_now[3:]) and not c.running
    c2 = ReviewController()
    messages = Recorder(c2.message)
    assert run(c2, record_of(GAME), fake("--crash-on-go", "3"), movetime_ms=20) is False
    assert messages.calls and not c2.running


@pytest.mark.skipif(not os.environ.get("CHESSGPU_ENGINE"), reason="the engine isn't built (CHESSGPU_ENGINE)")
def test_review_with_the_engine(qapp):
    c = ReviewController()
    r = record_of(GAME[:6])
    assert run(c, r, EngineProfile("ChessGPU", os.environ["CHESSGPU_ENGINE"], ["uci"]), movetime_ms=150) is True
    assert all(n.review is not None and n.review.depth >= 1 and n.review.pv for n in [r.root] + r.moves)
    assert all(j.kind is None for j in judge(r)[:2])  # 1. e4 e5: book moves, no mistakes


# --- Dialog, panel, window -------------------------------------------------------------------------------


def test_review_dialog_and_panel(qapp):
    d = ReviewDialog([fake(name="a"), fake(name="b")], "b", 41, {"speed": 0, "annotate": "false"})
    assert d.choice() == ("b", REVIEW_SPEEDS[0][1], None, False) and "41 positions" in d.estimate.text()
    d.speed.setCurrentIndex(3)
    d.annotate.setChecked(True)
    assert d.choice() == ("b", None, 12, True) and d.settings() == {"speed": 3, "annotate": "true"}
    panel = ReviewPanel()
    selected = Recorder(panel.node_selected)
    r = reviewed(["e4", "e5", "Nf3", "Nc6", "Bc4"], [20, 25, 50, -200, 120, 110])
    r.tags = {"White": "Ann", "Black": "Bo"}
    js = judge(r)
    panel.show_results(r, js, summarize(js))
    assert panel.cells[(WHITE, "name")].text() == "Ann" and panel.cells[(BLACK, "blunders")].text() == "1"
    assert panel.moves.count() == 2 and panel.moves.item(0).text().startswith("2. Nf3  Blunder")
    assert "h3 was best" in panel.moves.item(0).text()
    panel.moves.itemClicked.emit(panel.moves.item(1))
    assert selected.calls == [(r.moves[3],)]
    panel.set_running(True, 3, 11)
    assert panel.progress.isVisibleTo(panel) and "3 of 11" in panel.progress.format()
    panel.clear()
    assert panel.moves.count() == 0 and panel.cells[(WHITE, "accuracy")].text() == "–"


def test_review_in_the_window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json([fake("--blunder-at", "5", name="reviewer")]))
    w = MainWindow(settings)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    try:
        w._load(record_of(GAME))
        assert w.run_review("reviewer", movetime_ms=20)
        assert w.tabs.currentWidget() is w.review_panel and not w.review_action.isEnabled()
        assert wait_until(lambda: not w.review.running, 20_000)
        record = w.game.record
        bc4 = record.moves[4]
        assert bc4.nags == [4] and bc4.comment.startswith("Blunder. ") and len(bc4.parent.children) == 2
        assert w.review_panel.moves.count() == 1 and "Blunder" in w.review_panel.moves.item(0).text()
        assert w.graph.marks == {5: w.graph.marks[5]} and "Review done" in w.status.text()
        w.review_panel.moves.itemClicked.emit(w.review_panel.moves.item(0))
        assert w.view_node is bc4
        # Saved and opened again: the review comes back with the game.
        path = tmp_path / "reviewed.pgn"
        w.save_pgn_to(str(path))
        assert w.open_pgn_from(str(path))
        assert w.game.record is not record and w.review_panel.moves.count() == 1 and 5 in w.graph.marks
        # A new game in the middle of a review stops it.
        assert w.run_review("reviewer", movetime_ms=20)
        w.load_fen(START_FEN)
        assert not w.review.running and w.review_action.isEnabled()
    finally:
        w.close()
