"""Engine tournaments: the schedule, openings, Elo and standings,
adjudication, the controller with scripted engines (engines kept between
games, pause, stop, resume from the saved state, the PGN file), the dialog
and the window."""

import json
import math
import pathlib
import sys

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from chessgpu_gui.game import GameController, GameRecord, TimeControl
from chessgpu_gui.main_window import MainWindow
from chessgpu_gui.pgn import read_pgn
from chessgpu_gui.profiles import EngineProfile, profiles_to_json
from chessgpu_gui.rules import START_FEN, Position
from chessgpu_gui.score import MoveEval, Score
from chessgpu_gui.tournament import (Adjudication, Adjudicator, GameResult, TournamentController,
                                     TournamentSettings, crosstable, elo_of, elo_with_error, load_openings, los,
                                     schedule, standings)
from chessgpu_gui.tournament_panel import BUILT_IN_OPENINGS, TournamentDialog, TournamentPanel
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))


def fake(*flags, name="fake"):
    return EngineProfile(name, sys.executable, [FAKE, "--name", name, *flags])


# --- Schedule, openings, numbers ------------------------------------------------------------------


def test_schedule():
    games = schedule(["A", "B", "C"], "round robin", 2, 5)
    assert len(games) == 12 and [g.number for g in games] == list(range(1, 13))
    for e in "ABC":
        assert sum(g.white == e for g in games) == 4 and sum(g.black == e for g in games) == 4
    assert (games[0].white, games[0].black, games[1].white, games[1].black) == ("A", "B", "B", "A")
    assert games[0].opening == games[1].opening == 0 and games[2].opening == 1 and games[6].opening == 3
    assert schedule(["A", "B", "C"], "round robin", 1, 2)[4].opening == 0  # the openings start over
    gauntlet = schedule(["G", "X", "Y"], "gauntlet", 3, 100)
    assert len(gauntlet) == 12 and all("G" in (g.white, g.black) for g in gauntlet)
    assert schedule(["A", "B"], "round robin", 0, 1) == schedule(["A", "B"], "round robin", 1, 1)


def test_openings(tmp_path):
    assert load_openings("") == [Position.start()]
    built_in = load_openings(str(BUILT_IN_OPENINGS))
    assert len(built_in) == 200
    first = BUILT_IN_OPENINGS.read_text().splitlines()[1].split(";")[0].strip()
    assert built_in[0].fen() == first
    shuffled = load_openings(str(BUILT_IN_OPENINGS), random_order=True, seed=7)
    assert shuffled != built_in and sorted(p.fen() for p in shuffled) == sorted(p.fen() for p in built_in)
    assert shuffled == load_openings(str(BUILT_IN_OPENINGS), random_order=True, seed=7)
    epd = tmp_path / "o.epd"
    epd.write_text("# comment\nrnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 bm e5; id \"x\";\n\n")
    assert load_openings(str(epd))[0].fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    pgn = tmp_path / "o.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *\n\n1. d4 d5 *\n")
    lines = load_openings(str(pgn), plies=3)
    assert [p.fen().split()[0] for p in lines] == ["rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
                                                    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR"]
    for bad in ("8/8/8/8/8/8/8/8 w - - 0 1\n", "# nothing\n"):
        epd.write_text(bad)
        with pytest.raises(ValueError):
            load_openings(str(epd))
    pgn.write_text("1. e4 e5 2. Ke3 *")
    with pytest.raises(ValueError):
        load_openings(str(pgn))


def test_elo_and_los():
    assert elo_of(0.5) == 0 and elo_of(0.75) == pytest.approx(190.85, abs=0.01)
    assert elo_with_error(0, 0, 0) == (None, None) and elo_with_error(5, 0, 0) == (None, None)
    assert elo_with_error(0, 3, 4)[0] < 0 and elo_with_error(0, 10, 0) == (0.0, 0.0)
    elo, error = elo_with_error(10, 0, 10)
    margin = 1.959964 * math.sqrt(0.25 / 20)
    assert elo == 0 and error == pytest.approx((elo_of(0.5 + margin) - elo_of(0.5 - margin)) / 2)
    elo, error = elo_with_error(6, 2, 2)  # the trinomial variance: (6 * 0.3^2 + 2 * 0.2^2 + 2 * 0.7^2) / 10
    margin = 1.959964 * math.sqrt(0.16 / 10)
    assert elo == pytest.approx(elo_of(0.7)) and error == pytest.approx((elo_of(0.7 + margin) - elo_of(0.7 - margin)) / 2)
    assert los(0, 0) == 0.5 and los(5, 5) == 0.5 and los(10, 0) > 0.99 and los(0, 10) < 0.01


def results(*rows):
    return [GameResult(i + 1, w, b, r) for i, (w, b, r) in enumerate(rows)]


def test_standings_and_crosstable():
    rs = results(("A", "B", "1-0"), ("B", "A", "1/2-1/2"), ("A", "C", "0-1"), ("C", "A", "0-1"),
                 ("B", "C", "1-0"), ("C", "B", "*"))
    table = standings(["A", "B", "C"], rs)
    assert [(s.engine, s.games, s.wins, s.draws, s.losses, s.points) for s in table] == \
        [("A", 4, 2, 1, 1, 2.5), ("B", 3, 1, 1, 1, 1.5), ("C", 3, 1, 0, 2, 1.0)]
    assert table[0].score == pytest.approx(2.5 / 4) and table[0].elo()[0] == pytest.approx(elo_of(2.5 / 4))
    cross = crosstable(["A", "B", "C"], rs)
    assert cross[("A", "B")] == (1.5, 2) and cross[("B", "A")] == (0.5, 2) and cross[("A", "C")] == (1.0, 2)
    assert cross[("B", "C")] == (1.0, 1) and ("C", "B") in cross
    assert standings(["A", "B"], [])[0].games == 0


def evaluated(scores, fen=START_FEN):
    """A record whose moves carry the given reports (the mover's view, cp);
    the moves themselves: knights out and back."""
    r = GameRecord(Position.from_fen(fen))
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for i, cp in enumerate(scores):
        move = r.position.parse_uci(cycle[i % 4])
        r.push(move, MoveEval(Score(cp=cp), 10, 0.1) if cp is not None else None)
    return r


def test_adjudication():
    a = Adjudicator(Adjudication())
    # White ahead: White's engine says +650, Black's -700, three moves each.
    assert a.verdict(evaluated([50, -50, 650, -700, 640, -650, 700, -800])) == ("1-0", "White wins by adjudication")
    assert a.verdict(evaluated([650, -700, 640, -650, 700, 300])) is None  # Black's engine disagrees
    assert a.verdict(evaluated([-650, 700, -640, 650, -700, 800])) == ("0-1", "Black wins by adjudication")
    assert a.verdict(evaluated([650, -700, None, -650, 700, -800])) is None  # a move without a report
    late = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 40"
    assert a.verdict(evaluated([5, -5] * 8, late)) == ("1/2-1/2", "Draw by adjudication")
    assert a.verdict(evaluated([5, -5] * 8)) is None  # before move 34
    assert a.verdict(evaluated([5, -5] * 7 + [5, -30], late)) is None
    assert Adjudicator(Adjudication(max_moves=3)).verdict(evaluated([None] * 6)) == \
        ("1/2-1/2", "Draw by adjudication (move limit)")
    assert Adjudicator(Adjudication(enabled=False)).verdict(evaluated([650, -700] * 3)) is None


def test_settings_json():
    s = TournamentSettings(name="T", engines=["a", "b"], kind="gauntlet", rounds=3, tc={"movetime_ms": 50},
                           openings="x.epd", adjudication=Adjudication(resign_cp=900, max_moves=80))
    back = TournamentSettings.from_json(json.loads(json.dumps(s.to_json())))
    assert back == s and back.time_control() == TimeControl(movetime_ms=50)


# --- The controller -------------------------------------------------------------------------------------


def controller(profiles):
    game = GameController()
    by_name = {p.name: p for p in profiles}
    return game, TournamentController(game, by_name.get)


def test_tournament_plays_and_saves(qapp, tmp_path):
    game, t = controller([fake(name="A"), fake("--resign-none", name="B")])
    started, finished = Recorder(t.game_started), Recorder(t.finished)
    engines = []
    t.game_started.connect(lambda _p: engines.append(list(game.engines)))
    settings = TournamentSettings(name="Cup", engines=["A", "B"], rounds=2, tc={"movetime_ms": 20})
    pgn = tmp_path / "cup.pgn"
    t.start(settings, str(pgn))
    assert t.active and game.engine_cache is not None
    assert wait_until(lambda: finished.calls, 20_000)
    assert finished.calls == [(True,)] and not t.active and game.engine_cache is None
    assert [p.number for (p,) in started.calls] == [1, 2, 3, 4]
    assert [(r.white, r.black, r.result) for r in t.results] == [("A", "B", "1-0"), ("B", "A", "0-1")] * 2
    assert all("returned no move" in r.reason for r in t.results)
    table = t.standings()
    assert (table[0].engine, table[0].points, table[1].points) == ("A", 4, 0)
    # The engines stayed up: the second game's are the first's, colors swapped.
    assert engines[1][0] is engines[0][1] and engines[1][1] is engines[0][0]
    games = read_pgn(pgn.read_text(encoding="utf-8"))
    assert [g.tags["Round"] for g in games] == ["1", "2", "3", "4"] and all(g.tags["Event"] == "Cup" for g in games)
    assert [g.result for g in games] == ["1-0", "0-1", "1-0", "0-1"]
    state = json.loads(pgn.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(state["results"]) == 4 and state["settings"]["name"] == "Cup" and state["pgn"] == str(pgn)
    game.shutdown()


def test_tournament_adjudicates_pauses_stops_and_resumes(qapp, tmp_path):
    game, t = controller([fake(name="A"), fake(name="B")])
    settings = TournamentSettings(name="Long", engines=["A", "B"], rounds=2, tc={"movetime_ms": 10},
                                  adjudication=Adjudication(max_moves=4))
    pgn = tmp_path / "long.pgn"
    done = Recorder(t.game_finished)
    t.start(settings, str(pgn))
    t.pause()  # during game 1: it finishes, then the tournament waits
    assert t.pausing
    assert wait_until(lambda: t.state == t.PAUSED, 20_000)
    assert len(t.results) == 1 and t.results[0].result == "1/2-1/2" and "move limit" in t.results[0].reason
    assert t.results[0].plies == 8 and game.status == game.OVER
    t.resume()
    assert wait_until(lambda: t.current is not None and t.current.number == 2 and game.status == game.RUNNING)
    t.stop()  # game 2 aborted: not a result
    assert not t.active and len(t.results) == 1 and game.status == game.OVER
    assert len(done.calls) == 1
    settings_back, pgn_back, results_back = TournamentController.load_state(str(pgn.with_suffix(".json")))
    assert settings_back == settings and pgn_back == str(pgn) and len(results_back) == 1
    game2, t2 = controller([fake(name="A"), fake(name="B")])
    finished = Recorder(t2.finished)
    t2.start(settings_back, pgn_back, results_back)
    assert [p.number for p in t2.remaining()] == [2, 3, 4]
    assert wait_until(lambda: finished.calls, 20_000)
    assert sorted(r.number for r in t2.results) == [1, 2, 3, 4]
    assert len(read_pgn(pgn.read_text(encoding="utf-8"))) == 4
    with pytest.raises(ValueError):
        TournamentController.load_state(str(tmp_path / "missing.json"))
    with pytest.raises(ValueError):
        t2.start(TournamentSettings(engines=["A"]), str(pgn))  # one engine
    with pytest.raises(ValueError):
        t2.start(TournamentSettings(engines=["A", "nobody"]), str(pgn))
    game.shutdown()
    game2.shutdown()


def test_engines_kept_between_games(qapp, tmp_path):
    game, t = controller([fake(name="A"), fake(name="B"), fake(name="C")])
    lineups = []
    t.game_started.connect(lambda _p: lineups.append(list(game.engines)))
    finished = Recorder(t.finished)
    t.start(TournamentSettings(name="Three", engines=["A", "B", "C"], rounds=1, tc={"movetime_ms": 10},
                               adjudication=Adjudication(max_moves=2)), str(tmp_path / "three.pgn"))
    assert wait_until(lambda: finished.calls, 20_000) and len(lineups) == 6
    # Games: A-B, B-A, A-C, C-A, B-C, C-B. B sat out games 3 and 4 and came back in game 5: the same engine.
    b_first, b_back = lineups[0][1], lineups[4][0]
    assert b_back is b_first and b_first.state.value == "idle"
    assert len({id(e) for lineup in lineups for e in lineup}) == 3  # three engines in all
    assert not game.adjudicate("1-0", "too late")  # no game running: nothing to adjudicate
    assert game.record.reason == "Draw by adjudication (move limit)"
    game.shutdown()


# --- Dialog, panel, window ------------------------------------------------------------------------------


def test_tournament_dialog_and_panel(qapp, tmp_path):
    profiles = [fake(name="A"), fake(name="B"), fake(name="C")]
    d = TournamentDialog(profiles, {"engines": ["A", "B", "C"], "rounds": 2}, str(tmp_path))
    assert d.summary.text() == "3 engines, 12 games"
    d.kind.setCurrentIndex(1)
    d.gauntlet.setCurrentText("C")
    assert d.checked_engines() == ["C", "A", "B"] and d.summary.text() == "3 engines, 8 games"
    d.openings.setCurrentIndex(0)
    d.tc.mode.setCurrentText("Time per move")
    d.tc.movetime.setValue(0.05)
    d.max_moves.setValue(60)
    s = d.tournament()
    assert (s.kind, s.engines, s.rounds, s.openings, s.time_control()) == \
        ("gauntlet", ["C", "A", "B"], 2, "", TimeControl(movetime_ms=50))
    assert s.adjudication.max_moves == 60 and s.adjudication.enabled
    d.adjudicate.setChecked(False)
    assert not d.tournament().adjudication.enabled
    assert d.settings()["gauntlet"] == "C" and d.pgn.text().endswith("tournament.pgn")
    for i in range(3):
        d.engines.item(i).setCheckState(Qt.CheckState.Unchecked)
    d.kind.setCurrentIndex(0)
    d.accept()
    assert "two engines" in d.error.text() and d.result() == 0
    panel = TournamentPanel()
    rs = results(("A", "B", "1-0"), ("B", "A", "1/2-1/2"))
    panel.show_results(["A", "B"], standings(["A", "B"], rs), rs)
    assert panel.table.item(0, 0).text() == "A" and panel.table.item(0, 5).text() == "1.5"
    assert panel.cross.item(0, 1).text() == "1.5/2" and panel.cross.item(1, 0).text() == "0.5/2"
    assert panel.games.rowCount() == 2 and panel.games.item(1, 3).text() == "1/2-1/2"
    even = results(("A", "B", "1-0"), ("B", "A", "1-0"))  # one win each: 0 Elo, never "-0"
    panel.show_results(["A", "B"], standings(["A", "B"], even), even)
    assert panel.table.item(0, 7).text().startswith("0 ± ") and panel.table.item(1, 7).text().startswith("0 ± ")
    asked = Recorder(panel.game_requested)
    panel.games.cellDoubleClicked.emit(1, 2)
    assert asked.calls == [(2,)]
    panel.set_state("paused")
    assert panel.resume.isEnabled() and not panel.pause.isEnabled() and not panel.new.isEnabled()


def test_tournament_in_the_window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json([fake("--delay", "300", name="A"), fake("--resign-none", name="B")]))
    w = MainWindow(settings)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    try:
        s = TournamentSettings(name="Window cup", engines=["A", "B"], rounds=1, tc={"movetime_ms": 300})
        assert w.start_tournament(s, str(tmp_path / "games" / "cup.pgn"))
        assert w.tabs.currentWidget() is w.tournament_panel and not w.tournament_action.isEnabled()
        assert wait_until(lambda: w.game.status == w.game.RUNNING)
        assert w.game.record.tags["Event"] == "Window cup" and w.game.record.tags["Round"] == "1"
        assert "game 1 of 2" in w.tournament_panel.status.text()
        w.start_game(None, None, TimeControl())  # refused: the tournament plays
        assert w.game.record.tags.get("Event") == "Window cup" and "tournament" in w.status.text()
        assert not w.load_fen(START_FEN)
        assert wait_until(lambda: not w.tournament.active, 20_000)
        assert "finished, 2 of 2" in w.tournament_panel.status.text() and "Tournament over" in w.status.text()
        assert w.tournament_panel.table.item(0, 0).text() == "A" and w.tournament_panel.games.rowCount() == 2
        assert w.game.record.tags["Round"] == "2"  # the last game played is on the board
        w.tabs.setCurrentWidget(w.tournament_panel)
        w.show_tournament_game(1)
        assert w.game.record.tags["Round"] == "1" and w.game.record.result == "1-0" and w.tabs.currentIndex() == 0
        assert w.tournament_action.isEnabled()
        assert w.resume_tournament_from(str(tmp_path / "games" / "cup.json"))  # nothing left to play
        assert wait_until(lambda: not w.tournament.active)
        assert len(w.tournament.results) == 2
    finally:
        w.close()
