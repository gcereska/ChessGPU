"""Game records, clocks and the controllers, with humans (driven by the
test), the scripted fake engine and the real engine (CPU search)."""

import pathlib
import sys

import pytest

from chessgpu_gui.game import (AnalysisController, GameClock, GameController, GameRecord, TimeControl, ending_result,
                               termination_tag)
from chessgpu_gui.pgn import read_pgn, write_pgn
from chessgpu_gui.profiles import EngineProfile
from chessgpu_gui.rules import BLACK, START_FEN, WHITE, Ending, Move, Position
from chessgpu_gui.score import MoveEval, Score
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))


def fake(*flags, name="fake"):
    return EngineProfile(name, sys.executable, [FAKE, *flags])


def uci(text):
    return Move.from_uci(text)


# --- Time controls and clocks -----------------------------------------------------------


def test_time_control():
    tc = TimeControl(base_ms=300_000, inc_ms=2_000)
    assert tc.clocked and tc.text() == "300+2" and tc.pgn_tag() == "300+2"
    assert tc.engine_limits(1234.6, 99.4) == {"wtime": 1235, "btime": 99, "winc": 2000, "binc": 2000}
    assert tc.engine_limits(-5, 10)["wtime"] == 0
    assert TimeControl(movetime_ms=500).engine_limits(0, 0) == {"movetime": 500}
    assert TimeControl(depth=8, nodes=1000).engine_limits(0, 0) == {"depth": 8, "nodes": 1000}
    assert TimeControl().engine_limits(0, 0) == {"movetime": 1000}
    assert TimeControl().text() == "untimed" and TimeControl().pgn_tag() == "-"
    assert TimeControl(movetime_ms=1500, depth=5).text() == "1.5 s per move, depth 5"


def test_clock_with_fake_time():
    now = [100.0]
    clock = GameClock(TimeControl(base_ms=10_000, inc_ms=500), lambda: now[0])
    clock.start(WHITE)
    now[0] += 1.5
    assert clock.left(WHITE) == pytest.approx(8_500) and clock.left(BLACK) == 10_000
    assert clock.stop() == pytest.approx(1_500)
    clock.add_increment(WHITE)
    assert clock.remaining[WHITE] == pytest.approx(9_000)
    assert clock.stop() == 0.0 and clock.running is None


# --- Records -------------------------------------------------------------------------------


def record_of(*sans, fen=START_FEN):
    r = GameRecord(Position.from_fen(fen))
    for s in sans:
        r.push(r.position.parse_san(s))
    return r


def test_record_moves_positions_and_endings():
    r = record_of("f3", "e5", "g4", "Qh4#")
    assert [m.san for m in r.moves] == ["f3", "e5", "g4", "Qh4#"]
    assert len(r.positions) == 5 and r.position.is_checkmate()
    assert r.uci_moves() == ["f2f3", "e7e5", "g2g4", "d8h4"] and r.uci_moves(2) == ["f2f3", "e7e5"]
    assert r.ending() == ("0-1", "Black mates")
    r.pop()
    assert r.ending() is None and len(r.positions) == 4
    r.truncate(1)
    assert len(r.moves) == 1 and len(r.positions) == 2


def test_record_repetitions():
    r = record_of(*(["Nf3", "Nf6", "Ng1", "Ng8"] * 2))
    assert r.repetitions() == 3 and r.ending() == ("1/2-1/2", "Draw by threefold repetition")
    assert r.repetitions(4) == 2 and r.repetitions(0) == 1
    # A pawn move in between: earlier positions can't repeat.
    r2 = record_of("Nf3", "Nf6", "Ng1", "Ng8", "e3", "e6", "Nf3", "Nf6", "Ng1", "Ng8")
    assert r2.repetitions() == 2


def test_ending_results():
    pos = Position.start()
    assert ending_result(pos, Ending.STALEMATE) == ("1/2-1/2", "Draw by stalemate")
    assert ending_result(pos, Ending.FIFTY_MOVES)[1] == "Draw by the fifty-move rule"
    assert termination_tag("White loses on time") == "time forfeit"
    assert termination_tag("Black makes an illegal move: e2e5") == "rules infraction"
    assert termination_tag("White's engine failed (crashed)") == "abandoned"
    assert termination_tag("Black mates") == "normal"


def test_record_pgn_round_trip():
    r = record_of("e4", "e5", "Nf3")
    r.moves[1].eval = MoveEval(Score(cp=-20), 9, 0.25)
    r.moves[2].eval = MoveEval(None, None, 3.5)
    r.result, r.reason = "1-0", "Black resigns"
    r.tags = {"White": "me", "Black": "bot"}
    game = r.to_pgn()
    assert game.tags["Termination"] == "normal" and game.tags["PlyCount"] == "3"
    assert game.comments == ["", "-0.20/9 0.250s", "3.500s, Black resigns"]
    back = GameRecord.from_pgn(read_pgn(write_pgn(game))[0])
    assert [m.san for m in back.moves] == ["e4", "e5", "Nf3"] and back.result == "1-0"
    assert back.moves[1].eval == MoveEval(Score(cp=-20), 9, 0.25)
    assert back.tags["White"] == "me"


# --- Controller: humans ------------------------------------------------------------------------


def test_human_against_human(qapp):
    g = GameController()
    overs = Recorder(g.game_over)
    g.new_game(None, None, TimeControl())
    assert g.status == g.RUNNING and g.human_to_move
    assert not g.human_move(uci("e2e5"))
    for m in ("f2f3", "e7e5", "g2g4"):
        assert g.human_move(uci(m))
    assert g.human_move(uci("d8h4"))
    assert g.status == g.OVER and overs.calls == [("0-1", "Black mates")]
    assert not g.human_move(uci("a2a3"))
    assert g.record.moves[0].eval.seconds is not None
    assert g.record.tags["White"] == "Human" and g.record.tags["TimeControl"] == "-"


def test_resign_and_abort(qapp):
    g = GameController()
    g.new_game(None, None, TimeControl())
    g.human_move(uci("e2e4"))
    g.resign()
    assert (g.record.result, g.record.reason) == ("1-0", "Black resigns")
    g.new_game(None, None, TimeControl())
    g.abort()
    assert (g.record.result, g.record.reason, g.status) == ("*", "Game aborted", g.OVER)


def test_takeback_human_against_human(qapp):
    g = GameController()
    g.new_game(None, None, TimeControl())
    g.human_move(uci("e2e4"))
    g.human_move(uci("e7e5"))
    assert g.takeback() and len(g.record.moves) == 1 and g.record.position.turn == BLACK


# --- Controller: engines ------------------------------------------------------------------------


@pytest.fixture
def controller(qapp):
    made = []

    def make():
        g = GameController()
        g.overs = Recorder(g.game_over)
        g.infos = Recorder(g.engine_info)
        made.append(g)
        return g

    yield make
    for g in made:
        g.shutdown()


def over(g, timeout_ms=30_000):
    return wait_until(lambda: g.status == g.OVER, timeout_ms)


def test_human_against_engine_and_takeback(controller):
    g = controller()
    g.new_game(None, fake(), TimeControl(movetime_ms=50))
    assert wait_until(lambda: g.status == g.RUNNING)
    assert g.human_to_move
    assert g.human_move(uci("e2e4"))
    assert not g.human_to_move and g.thinking_color == BLACK
    assert wait_until(lambda: len(g.record.moves) == 2)
    reply = g.record.moves[1]
    assert reply.move == uci("a7a5") and reply.eval.score == Score(cp=15) and reply.eval.depth == 2
    assert g.infos.calls and all(c[0] == BLACK for c in g.infos.calls)
    assert g.human_to_move
    assert g.takeback()
    assert len(g.record.moves) == 0 and g.human_to_move


def test_engine_starts_when_it_plays_white(controller):
    g = controller()
    g.new_game(fake(), None, TimeControl(depth=1))
    assert wait_until(lambda: len(g.record.moves) == 1)
    assert g.record.moves[0].move == uci("a2a3") and g.human_to_move
    assert g.takeback()  # the engine's first move goes; the engine moves again
    assert wait_until(lambda: len(g.record.moves) == 1 and g.human_to_move)


def test_engine_against_engine_plays_to_the_end(controller):
    g = controller()
    g.new_game(fake(name="one"), fake(name="two"), TimeControl(depth=1))
    assert over(g, 60_000)
    assert g.record.result in ("1-0", "0-1", "1/2-1/2") and g.record.reason
    assert g.record.ending() == (g.record.result, g.record.reason)
    assert not g.takeback()  # no human in this game


def test_illegal_engine_move_loses(controller):
    g = controller()
    g.new_game(None, fake("--illegal"), TimeControl(movetime_ms=50))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("d2d4"))
    assert over(g)
    assert g.record.result == "1-0" and "Black makes an illegal move" in g.record.reason
    assert len(g.record.moves) == 1


def test_engine_without_a_move_loses(controller):
    g = controller()
    g.new_game(fake("--resign-none"), None, TimeControl(depth=1))
    assert over(g)
    assert (g.record.result, g.record.reason) == ("0-1", "White returned no move")


def test_time_forfeit(controller):
    g = controller()
    g.new_game(fake("--wait-for-stop"), None, TimeControl(base_ms=300))
    assert over(g, 10_000)
    assert (g.record.result, g.record.reason) == ("0-1", "White loses on time")
    # The engine's move after the stop arrives late and is ignored.
    assert not wait_until(lambda: g.record.moves, timeout_ms=300)


def test_time_forfeit_against_a_lone_king_draws(controller):
    g = controller()
    g.new_game(fake("--wait-for-stop"), None, TimeControl(base_ms=300),
               start=Position.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"))
    assert over(g, 10_000)
    assert g.record.result == "1/2-1/2" and "cannot mate" in g.record.reason


def test_clock_starts_after_engine_start_up(controller):
    g = controller()
    g.new_game(fake("--slow-start", "0.8"), None, TimeControl(base_ms=2_000, inc_ms=100))
    assert wait_until(lambda: g.record.moves, 15_000)
    left = g.record.moves[0].clock_ms
    assert left is not None and left > 1_900  # the 0.8 s start-up wasn't charged


def test_takeback_restores_clocks(controller):
    g = controller()
    g.new_game(None, fake("--delay", "100"), TimeControl(base_ms=60_000, inc_ms=1_000))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("e2e4"))
    assert wait_until(lambda: len(g.record.moves) == 2)
    g.human_move(uci("d2d4"))
    assert wait_until(lambda: len(g.record.moves) == 4)
    after_first = g.record.moves[0].clock_ms
    assert g.takeback()
    assert len(g.record.moves) == 2 and g.clock.remaining[WHITE] == pytest.approx(after_first)
    assert g.clock.remaining[BLACK] == pytest.approx(g.record.moves[1].clock_ms)


def test_takeback_while_the_engine_thinks(controller):
    g = controller()
    g.new_game(None, fake("--wait-for-stop"), TimeControl(movetime_ms=60_000))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("e2e4"))
    assert wait_until(lambda: g.thinking_color == BLACK)
    assert g.takeback()
    # The stopped search's bestmove arrives and must not be played.
    assert wait_until(lambda: not g.engines[BLACK].searching)
    assert not wait_until(lambda: g.record.moves, timeout_ms=300)
    assert g.human_to_move and g.record.position == Position.start()


def test_stale_bestmove_after_takeback_and_a_new_move(controller):
    """The stopped search's bestmove arrives while the engine's next search
    (after the human's new move) is already queued: it must be ignored."""
    g = controller()
    g.new_game(None, fake("--wait-for-stop"), TimeControl(movetime_ms=60_000))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("e2e4"))
    assert wait_until(lambda: g.thinking_color == BLACK)
    assert g.takeback()
    assert g.human_move(uci("d2d4"))  # before the stopped search's bestmove is processed
    assert not wait_until(lambda: len(g.record.moves) > 1, timeout_ms=400)
    assert g.thinking_color == BLACK
    g.move_now()
    assert wait_until(lambda: len(g.record.moves) == 2)
    assert [m.san for m in g.record.moves] == ["d4", "a5"]


def test_clock_arithmetic_with_increment(qapp):
    now = [0.0]
    g = GameController(now=lambda: now[0])
    g.new_game(None, None, TimeControl(base_ms=1_000, inc_ms=500))
    now[0] += 0.2
    g.human_move(uci("e2e4"))
    assert g.clock.remaining[WHITE] == pytest.approx(1_300) and g.record.moves[0].clock_ms == pytest.approx(1_300)
    assert g.record.moves[0].eval.seconds == pytest.approx(0.2)
    now[0] += 0.7
    assert g.clock.left(BLACK) == pytest.approx(300)
    g.human_move(uci("e7e5"))
    assert g.clock.remaining[BLACK] == pytest.approx(800)
    g.shutdown()


def test_late_move_loses_on_time(qapp):
    now = [0.0]
    g = GameController(now=lambda: now[0])
    g.new_game(None, None, TimeControl(base_ms=1_000, inc_ms=5_000))
    assert g.human_move(uci("e2e4"))
    now[0] += 1.2  # black thinks 1.2 s of a 1 s clock; the increment can't save the move
    g.human_move(uci("e7e5"))
    assert g.status == g.OVER and (g.record.result, g.record.reason) == ("1-0", "Black loses on time")
    assert len(g.record.moves) == 1
    g.shutdown()


def test_move_now(controller):
    g = controller()
    g.new_game(None, fake("--wait-for-stop"), TimeControl(movetime_ms=60_000))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("e2e4"))
    assert not wait_until(lambda: len(g.record.moves) == 2, timeout_ms=300)
    g.move_now()
    assert wait_until(lambda: len(g.record.moves) == 2)


def test_engine_crash_loses(controller):
    g = controller()
    g.new_game(None, fake("--crash-on-go", "2"), TimeControl(movetime_ms=20))
    assert wait_until(lambda: g.status == g.RUNNING)
    g.human_move(uci("e2e4"))
    assert wait_until(lambda: len(g.record.moves) == 2)
    g.human_move(uci("d2d4"))
    assert over(g)
    assert g.record.result == "1-0" and "Black's engine failed" in g.record.reason


def test_engine_that_cannot_start(controller):
    g = controller()
    g.new_game(EngineProfile("missing", str(pathlib.Path(FAKE).with_name("missing.exe"))), None, TimeControl())
    assert over(g)
    assert g.record.result == "*" and g.record.reason.startswith("Game not started")


def test_engines_are_reused_between_games(controller):
    g = controller()
    g.new_game(None, fake(), TimeControl(depth=1))
    assert wait_until(lambda: g.status == g.RUNNING)
    first = g.engines[BLACK]
    g.new_game(None, fake(), TimeControl(depth=1))
    assert g.engines[BLACK] is first
    assert wait_until(lambda: g.status == g.RUNNING)
    g.new_game(None, fake("--delay", "1"), TimeControl(depth=1))  # different options: a new process
    assert g.engines[BLACK] is not first


def test_real_engines_play_a_game(controller, engine):
    cpu = EngineProfile("ChessGPU cpu", str(engine), ["uci"], {"Search": "cpu"})
    g = controller()
    g.new_game(cpu, cpu, TimeControl(depth=2))
    assert wait_until(lambda: g.status == g.OVER or len(g.record.moves) >= 120, 120_000)
    if g.status != g.OVER:
        g.abort()
    assert len(g.record.moves) >= 20
    assert all(m.eval.depth == 2 for m in g.record.moves if m.eval and m.eval.depth)


# --- Analysis --------------------------------------------------------------------------------------


def test_analysis(qapp):
    a = AnalysisController()
    infos = Recorder(a.info)
    try:
        a.set_profile(fake("--wait-for-stop"))
        sent = []
        a.engine.log.connect(lambda d, line: sent.append(line) if d == ">" else None)
        a.analyze(START_FEN, [])
        assert wait_until(lambda: infos.calls)
        assert infos.calls[0][0].pv == ["a2a3"]
        # One search, nothing stopped: the analysis started exactly once.
        assert not wait_until(lambda: "stop" in sent, timeout_ms=300)
        assert sum(1 for line in sent if line.startswith("go")) == 1
        switch = len(infos.calls)
        a.analyze(START_FEN, ["e2e4"])
        assert wait_until(lambda: any(c[0].pv == ["a7a5"] for c in infos.calls))
        assert not wait_until(lambda: any(c[0].pv != ["a7a5"] for c in infos.calls[switch:]), timeout_ms=300)
        a.stop()
        assert not a.running
        a.analyze("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", [])  # stalemate: nothing to analyse
        assert not a.running
    finally:
        a.shutdown()
