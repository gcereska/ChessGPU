"""The UCI client: parsing, and UciEngine against a scripted fake engine
(tests/fake_engine.py) and against the real engine on its CPU search."""

import pathlib
import sys

import pytest

from chessgpu_gui.rules import START_FEN, Position
from chessgpu_gui.score import Score
from chessgpu_gui.uci import EngineState, Info, UciEngine, UciOption, go_command, parse_bestmove, parse_info, parse_option
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))


# --- Parsing -------------------------------------------------------------------------


@pytest.mark.parametrize("line,expected", [
    ("option name Hash type spin default 16 min 1 max 1024", UciOption("Hash", "spin", "16", 1, 1024)),
    ("option name Clear Hash type button", UciOption("Clear Hash", "button")),
    ("option name Ponder type check default false", UciOption("Ponder", "check", "false")),
    ("option name Style type combo default Normal var Normal var Wild var Calm Mode",
     UciOption("Style", "combo", "Normal", vars=["Normal", "Wild", "Calm Mode"])),
    ("option name Book File type string default <empty>", UciOption("Book File", "string", "")),
    ("option name Debug Log File type string default", UciOption("Debug Log File", "string", "")),
    ("option name NodeBudget type spin default 2048 min 0 max 100000000", UciOption("NodeBudget", "spin", "2048", 0,
                                                                                   100000000)),
    ("option name Search type combo default ybwc-device var cpu var gpu-independent var ybwc-host var ybwc-device",
     UciOption("Search", "combo", "ybwc-device", vars=["cpu", "gpu-independent", "ybwc-host", "ybwc-device"])),
    ("option name Odd min type spin default 1 min 0 max 9", UciOption("Odd min", "spin", "1", 0, 9)),
])
def test_parse_option(line, expected):
    assert parse_option(line) == expected


@pytest.mark.parametrize("line", ["option", "option type spin", "option name X", "foo name X type spin"])
def test_parse_option_rejects(line):
    assert parse_option(line) is None


def test_option_normalize():
    spin = UciOption("Hash", "spin", "16", 1, 1024)
    assert spin.normalize(64) == "64" and spin.normalize(" 7 ") == "7"
    assert spin.normalize(0) is None and spin.normalize(2000) is None and spin.normalize("x") is None
    combo = UciOption("Style", "combo", "Normal", vars=["Normal", "Calm Mode"])
    assert combo.normalize("calm mode") == "Calm Mode" and combo.normalize("loud") is None
    check = UciOption("Ponder", "check", "false")
    assert check.normalize(True) == "true" and check.normalize("FALSE") == "false" and check.normalize("1") is None
    assert UciOption("Path", "string").normalize(" a b ") == "a b"


def test_parse_info():
    ours = parse_info("info depth 8 score cp 7 nodes 71703841 nps 125580000 time 571 pv e2e4 d7d5 e4d5")
    assert ours == Info(depth=8, score=Score(cp=7), nodes=71703841, nps=125580000, time_ms=571,
                        pv=["e2e4", "d7d5", "e4d5"])
    sf = parse_info("info depth 24 seldepth 33 multipv 1 score cp -15 upperbound wdl 30 900 70 nodes 123 nps 4 "
                    "hashfull 211 tbhits 0 time 30 pv g1f3 d7d5 b1c3")
    assert (sf.seldepth, sf.multipv, sf.score, sf.bound, sf.hashfull, sf.tbhits) == (33, 1, Score(cp=-15), "upper",
                                                                                      211, 0)
    assert sf.pv == ["g1f3", "d7d5", "b1c3"]
    mate = parse_info("info depth 3 score mate -2 lowerbound nodes 10 pv e7e8q a1a2")
    assert mate.score == Score(mate=-2) and mate.bound == "lower" and mate.pv == ["e7e8q", "a1a2"]
    assert parse_info("info string search ybwc-device quad/pesto").string == "search ybwc-device quad/pesto"
    cur = parse_info("info depth 5 currmove e2e4 currmovenumber 1")
    assert cur.currmove == "e2e4" and cur.currmovenumber == 1 and cur.score is None
    bad = parse_info("info depth x score cp y nodes 5 pv")
    assert bad.depth is None and bad.score is None and bad.nodes == 5 and bad.pv == []


def test_parse_bestmove_and_go_command():
    assert parse_bestmove("bestmove e2e4") == ("e2e4", None)
    assert parse_bestmove("bestmove e7e8q ponder a1a2") == ("e7e8q", "a1a2")
    assert parse_bestmove("bestmove (none)") == (None, None)
    assert parse_bestmove("bestmove 0000") == (None, None)
    assert go_command({"wtime": 1000.4, "btime": 900, "winc": 10, "binc": 10}) == "go wtime 1000 btime 900 winc 10 binc 10"
    assert go_command({"depth": 8}) == "go depth 8"
    assert go_command({"infinite": True, "movetime": None}) == "go infinite"
    assert go_command({"movetime": 500}) == "go movetime 500"


# --- UciEngine against the fake engine ---------------------------------------------------


@pytest.fixture
def make_engine(qapp):
    engines = []

    def make(*flags, options=None, program=None, args=None, **kwargs):
        e = UciEngine(kwargs.pop("name", "fake"), **kwargs)
        e.lines = []
        e.log.connect(lambda d, line: e.lines.append((d, line)))
        e.failures = Recorder(e.failed)
        e.bests = Recorder(e.bestmove)
        e.infos = Recorder(e.info)
        e.start(program or sys.executable, args if args is not None else [FAKE, *flags], options or {})
        engines.append(e)
        return e

    yield make
    for e in engines:
        e.quit()


def sent(e):
    return [line for d, line in e.lines if d == ">"]


def ready(e, timeout_ms=10_000):
    return wait_until(lambda: e.state == EngineState.IDLE or e.failures.calls, timeout_ms) and not e.failures.calls


def test_handshake_and_options(make_engine):
    e = make_engine()
    assert ready(e)
    assert e.id_name == "Fake" and e.id_author == "the tests"
    assert [o.name for o in e.options] == ["Hash", "Style", "Ponder", "Clear Hash", "Book File"]
    assert e.option("hash").max == 1024 and e.option("style").vars == ["Normal", "Wild", "Calm Mode"]
    assert sent(e)[:2] == ["uci", "isready"]


def test_startup_options_are_validated(make_engine):
    e = make_engine(options={"Hash": 64, "style": "calm mode", "NoSuch": 1, "Ponder": "maybe", "Clear Hash": True})
    assert ready(e)
    assert sent(e) == ["uci", "setoption name Hash value 64", "setoption name Style value Calm Mode",
                       "setoption name Clear Hash", "isready"]
    events = [line for d, line in e.lines if d == "!"]
    assert any("NoSuch" in line for line in events) and any("Ponder" in line for line in events)


def test_search_ids_and_results(make_engine):
    e = make_engine()
    assert ready(e)
    sid = e.go(START_FEN, depth=1)
    assert wait_until(lambda: e.bests.calls)
    assert e.bests.calls == [(sid, "a2a3", None)]
    assert [c[0] for c in e.infos.calls] == [sid, sid]
    assert e.infos.calls[0][1].score == Score(cp=12) and e.infos.calls[0][1].pv == ["a2a3"]
    sid2 = e.go(START_FEN, ["e2e4"], movetime=100)
    assert sid2 == sid + 1
    assert wait_until(lambda: len(e.bests) == 2)
    assert e.bests.calls[1] == (sid2, "a7a5", None)
    assert sent(e)[-2:] == ["position fen " + START_FEN + " moves e2e4", "go movetime 100"]
    assert e.state == EngineState.IDLE and not e.searching


def test_infinite_search_runs_until_stop(make_engine):
    e = make_engine()
    assert ready(e)
    sid = e.go(START_FEN, infinite=True)
    assert wait_until(lambda: e.infos.calls)
    assert not wait_until(lambda: e.bests.calls, timeout_ms=300)
    assert e.state == EngineState.SEARCHING
    e.stop()
    assert wait_until(lambda: e.bests.calls)
    assert e.bests.calls[0][0] == sid


def test_new_search_waits_for_the_old_bestmove(make_engine):
    e = make_engine()
    assert ready(e)
    first = e.go(START_FEN, infinite=True)
    second = e.go(START_FEN, ["d2d4"], depth=1)
    assert wait_until(lambda: len(e.bests) == 2)
    assert [c[0] for c in e.bests.calls] == [first, second]
    assert e.bests.calls[1][1] == "a7a5"
    assert sent(e)[-5:] == ["position fen " + START_FEN, "go infinite", "stop",
                            "position fen " + START_FEN + " moves d2d4", "go depth 1"]


def test_reentrant_go_from_a_state_listener(make_engine):
    """A listener that starts another search when the first one starts: the
    stop must reach the engine after the go it stops, never before."""
    e = make_engine()
    assert ready(e)
    second = []

    def on_state(state):
        if state == EngineState.SEARCHING and not second:
            second.append(e.go(START_FEN, ["e2e4"], depth=1))

    e.state_changed.connect(on_state)
    first = e.go(START_FEN, infinite=True)
    assert second, "the listener didn't run"
    assert wait_until(lambda: len(e.bests) == 2)
    assert [c[0] for c in e.bests.calls] == [first, second[0]]
    assert sent(e)[-5:] == ["position fen " + START_FEN, "go infinite", "stop",
                            "position fen " + START_FEN + " moves e2e4", "go depth 1"]
    assert e.state == EngineState.IDLE


def test_stop_cancels_a_queued_search(make_engine):
    e = make_engine()
    assert ready(e)
    first = e.go(START_FEN, infinite=True)
    e.go(START_FEN, depth=1)  # queued behind the first, then cancelled
    e.stop()
    assert wait_until(lambda: e.bests.calls)
    assert not wait_until(lambda: len(e.bests) > 1, timeout_ms=300)
    assert [c[0] for c in e.bests.calls] == [first]
    assert sent(e).count("go depth 1") == 0 and e.state == EngineState.IDLE


def test_commands_wait_for_the_search_to_end(make_engine):
    e = make_engine()
    assert ready(e)
    e.go(START_FEN, infinite=True)
    assert e.set_option("Hash", 128)
    e.new_game()
    assert wait_until(lambda: e.bests.calls)
    log = sent(e)
    assert log.index("stop") < log.index("setoption name Hash value 128") < log.index("ucinewgame")
    assert not e.set_option("Hash", 5000) and not e.set_option("Style", "loud") and not e.set_option("Nope", 1)
    assert e.set_option("Clear Hash", None)
    assert wait_until(lambda: "setoption name Clear Hash" in sent(e))


def test_no_legal_move_answer(make_engine):
    e = make_engine("--resign-none")
    assert ready(e)
    sid = e.go(START_FEN, depth=1)
    assert wait_until(lambda: e.bests.calls)
    assert e.bests.calls == [(sid, None, None)]


def test_crash_is_reported(make_engine):
    e = make_engine("--crash-on-go", "1")
    assert ready(e)
    e.go(START_FEN, depth=1)
    assert wait_until(lambda: e.failures.calls)
    assert e.state == EngineState.FAILED
    assert "exited unexpectedly" in e.failures.calls[0][0] or "crashed" in e.failures.calls[0][0]
    with pytest.raises(RuntimeError):
        e.go(START_FEN, depth=1)


def test_handshake_timeouts(make_engine):
    e = make_engine("--no-uciok", uciok_timeout_ms=500)
    assert wait_until(lambda: e.failures.calls)
    assert "no uciok within 0.5 s" in e.failures.calls[0][0] and e.state == EngineState.FAILED
    e2 = make_engine("--no-readyok", readyok_timeout_ms=500)
    assert wait_until(lambda: e2.failures.calls)
    assert "no readyok" in e2.failures.calls[0][0]


def test_missing_program(make_engine):
    e = make_engine(program=str(pathlib.Path(FAKE).with_name("no_such_engine.exe")), args=[])
    assert wait_until(lambda: e.failures.calls)
    assert "failed to start" in e.failures.calls[0][0]


def test_quit(make_engine):
    e = make_engine()
    assert ready(e)
    e.go(START_FEN, infinite=True)
    e.quit()
    assert e.state == EngineState.STOPPED and not e.failures.calls
    e.quit()  # twice is harmless


# --- UciEngine against the real engine (CPU search: no GPU needed) ---------------------------


def test_real_engine(make_engine, engine):
    e = make_engine(program=str(engine), args=["uci"], options={"Search": "cpu"})
    assert ready(e)
    assert e.id_name == "ChessGPU"
    names = {o.name for o in e.options}
    assert {"Search", "Board", "Model", "Eval", "Ordering", "Stack", "LeafDepth", "NodeBudget"} <= names
    assert e.option("Search").vars == ["cpu", "gpu-independent", "ybwc-host", "ybwc-device", "abdada-cpu",
                                       "abdada-device"]
    assert e.option("Eval").vars == ["pesto", "material", "terms", "nnue"]
    sid = e.go(START_FEN, depth=4)
    assert wait_until(lambda: e.bests.calls)
    depths = [c[1].depth for c in e.infos.calls if c[1].depth]
    assert depths == [1, 2, 3, 4]
    assert all(c[1].score is not None and c[1].pv for c in e.infos.calls if c[1].depth)
    best = e.bests.calls[0]
    assert best[0] == sid and Position.start().parse_uci(best[1])
    # Infinite search, stopped.
    sid = e.go(START_FEN, ["e2e4"], infinite=True)
    assert wait_until(lambda: len(e.infos.calls) > 6)
    e.stop()
    assert wait_until(lambda: len(e.bests) == 2)
    assert e.bests.calls[1][0] == sid
    Position.start().push(Position.start().parse_uci("e2e4")).parse_uci(e.bests.calls[1][1])


def test_stockfish_as_an_independent_engine(make_engine):
    """Our client against an engine we didn't write (skipped without it)."""
    from chessgpu_gui.profiles import STOCKFISH
    if not STOCKFISH.exists():
        pytest.skip("Stockfish not in tools/")
    e = make_engine(program=str(STOCKFISH), args=[], options={"Threads": 1, "Hash": 16})
    assert ready(e)
    assert e.id_name.startswith("Stockfish")
    assert e.option("Hash").type == "spin" and e.option("Clear Hash").type == "button"
    assert e.option("Threads").min == 1 and e.option("MultiPV") is not None
    sid = e.go(START_FEN, ["e2e4"], depth=8)
    assert wait_until(lambda: e.bests.calls, 30_000)
    scored = [c[1] for c in e.infos.calls if c[1].score is not None and c[1].depth]
    assert scored and scored[-1].depth == 8 and scored[-1].pv
    best = e.bests.calls[0]
    assert best[0] == sid
    after = Position.start().push(Position.start().parse_uci("e2e4"))
    after.parse_uci(best[1])  # a legal reply
