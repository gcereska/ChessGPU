"""Game databases: scanning PGN files, reading main lines quickly, the
position index (in this process and in worker processes), search, the
opening book in games, the explorer and database panels, and the window."""

import pathlib
import random
import sys

import pytest
from PySide6.QtCore import QModelIndex, QSettings, Qt
from PySide6.QtWidgets import QApplication

from chessgpu_gui.database import (DbGame, GameDatabase, PositionIndex, explorer_key, index_games, mainline,
                                   scan_pgn)
from chessgpu_gui.database_panel import DatabasePanel, ExplorerPanel
from chessgpu_gui.game import GameController, TimeControl
from chessgpu_gui.main_window import MainWindow
from chessgpu_gui.pgn import PgnError
from chessgpu_gui.profiles import EngineProfile, profiles_to_json
from chessgpu_gui.rules import Move, Position
from qt_helpers import Recorder, wait_until

FAKE = str(pathlib.Path(__file__).with_name("fake_engine.py"))
DATA = pathlib.Path(__file__).resolve().parent / "data"

GAMES = """[Event "Club"] [Site "Here"]
[White "Ann"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 {a comment} Nc6 (2... d6 3. d4) 3. Bb5 a6 $1 1-0

% an escape line
[Event "Club"]
[White "Cy"]
[Black "Ann \\"the\\" second"]
[Result "1/2-1/2"]

1. e4 c5 2. Nf3 d6 1/2-1/2

[Event "Open"]
[White "Bob"]
[Black "Cy"]
[Result "0-1"]

1. d4 d5 2. c4 0-1

[Event "Open"]
[White "Dee"]
[Black "Bob"]
[Result "1-0"]
[SetUp "1"]
[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]

1. e4 Kd7 1-0
"""


def fake(*flags, name="fake"):
    return EngineProfile(name, sys.executable, [FAKE, *flags])


def start_after(*uci):
    pos = Position.start()
    for u in uci:
        pos = pos.push(Move.from_uci(u))
    return pos


# --- Scanning and indexing -----------------------------------------------------------------------


def test_scan_pgn():
    games = scan_pgn(GAMES, first_number=5)
    assert [g.number for g in games] == [5, 6, 7, 8]
    assert games[0].tags == {"Event": "Club", "Site": "Here", "White": "Ann", "Black": "Bob", "Result": "1-0"}
    assert games[1].tag("Black") == 'Ann "the" second' and games[1].result == "1/2-1/2"
    assert "escape line" not in games[1].text and games[3].tag("FEN").startswith("4k3")
    record = games[0].record()
    assert [n.san for n in record.moves] == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert record.moves[3].parent.children[1].san == "d6"  # the variation came along
    assert DbGame(1, {"Result": "2-0"}, "").result == "*"
    with pytest.raises(PgnError):
        DbGame(1, {}, "1. e4 e5 2. Ke3 *").record()
    assert scan_pgn("") == [] and len(scan_pgn("1. e4 e5 *")) == 1


def test_mainline_reads_fast_and_skips_the_rest():
    start, moves = mainline(scan_pgn(GAMES)[0].text, 40)
    assert start == Position.start() and [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    assert len(mainline(scan_pgn(GAMES)[0].text, 3)[1]) == 3
    start, moves = mainline(scan_pgn(GAMES)[3].text, 40)
    assert start.fen() == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" and [m.uci() for m in moves] == ["e2e4", "e8d7"]
    with pytest.raises(PgnError):
        mainline("1. e4 e5 2. Ke3 *", 10)
    with pytest.raises(PgnError):
        mainline('[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n1. e4 *', 10)


def test_position_index():
    games = scan_pgn(GAMES)
    items = [(g.number, g.result, g.text) for g in games] + [(9, "*", "1. e4 e5 2. Ke3 *")]
    index = index_games(items, 40)
    assert index.indexed == 4 and index.failed == 1
    stats = dict((m.uci(), s) for m, s in index.stats(Position.start()))
    assert stats == {"e2e4": [1, 1, 0, 0], "d2d4": [0, 0, 1, 0]}
    assert [m.uci() for m, _ in index.stats(Position.start())] == ["e2e4", "d2d4"]  # most played first
    assert index.games_at(Position.start()) == [1, 2, 3]
    assert index.games_at(start_after("e2e4", "e7e5", "g1f3")) == [1]
    assert index.games_at(start_after("d2d4", "d7d5", "c2c4")) == [3]  # the last position counts too
    assert index.stats(start_after("e2e4", "e7e5", "g1f3")) == [(Move.from_uci("b8c6"), [1, 0, 0, 0])]
    short = index_games(items, 2)
    assert short.games_at(start_after("e2e4", "e7e5", "g1f3")) == [] and short.games_at(start_after("e2e4", "e7e5"))
    # Merging chunks gives the whole.
    merged = PositionIndex(40)
    merged.merge(index_games(items[:2], 40))
    merged.merge(index_games(items[2:], 40))
    assert merged.moves == index.moves and sorted(merged.games) == sorted(index.games)
    assert {k: sorted(v) for k, v in merged.games.items()} == {k: sorted(v) for k, v in index.games.items()}
    # A book move: chosen in proportion to how often it was played.
    rng = random.Random(3)
    picks = [index.book_move(Position.start(), rng).uci() for _ in range(600)]
    assert set(picks) == {"e2e4", "d2d4"} and 330 < picks.count("e2e4") < 470
    assert index.book_move(start_after("a2a3"), rng) is None
    assert index.book_move(Position.start(), rng, min_games=3) is None
    assert explorer_key(Position.start()) == explorer_key(Position.from_fen(Position.start().fen()))


def test_database_search_and_index(qapp, tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(GAMES, encoding="utf-8")
    db = GameDatabase()
    changed = Recorder(db.changed)
    assert db.add_file(str(path)) == 4 and db.name == "games.pgn" and changed.calls
    assert db.add_file(str(path)) == 4 and len(db.games) == 8 and db.name == "games.pgn and 1 more"
    assert [g.number for g in db.search("ann")] == [1, 2, 5, 6]
    assert [g.number for g in db.search("", "0-1")] == [3, 7]
    assert [g.number for g in db.search("bob", "1-0")] == [1, 4, 5, 8]
    assert db.search(position=Position.start()) == []  # no index yet
    progress = Recorder(db.index_progress)
    db.build_index(workers=0, wait=True)
    assert db.index.indexed == 8 and progress.calls[-1] == (8, 8) and not db.building
    assert [g.number for g in db.search(position=start_after("e2e4", "e7e5"))] == [1, 5]
    assert db.game(8).tag("White") == "Dee" and db.game(9) is None and db.game(0) is None
    single = db.index
    db.build_index(workers=2, wait=True, chunk=3)  # in worker processes: the same index
    assert db.index.moves == single.moves
    ready = Recorder(db.index_ready)
    db.build_index()  # in the background
    assert wait_until(lambda: ready.calls, 20_000) and db.index.indexed == 8
    db.clear()
    assert db.games == [] and db.index is None and db.name == ""


def test_new_game_book_choice(qapp):
    from chessgpu_gui.dialogs import NewGameDialog

    d = NewGameDialog([fake(name="bot")], Position.start(), {}, None, "games.pgn")
    assert d.book.isEnabled() and not d.book.isChecked() and d.choice().book_plies == 0
    d.book.setChecked(True)
    d.book_moves.setValue(6)
    assert d.choice().book_plies == 12 and d.settings()["book"] == "true"
    none = NewGameDialog([fake(name="bot")], Position.start(), {"book": "true"}, None, "")
    assert not none.book.isEnabled() and none.choice().book_plies == 0


def test_book_in_a_game(qapp):
    index = index_games([(g.number, g.result, g.text) for g in scan_pgn(GAMES)], 40)
    g = GameController()
    g.book, g.book_plies = index, 4
    g.rng = random.Random(1)
    g.new_game(fake(), fake(), TimeControl(movetime_ms=10))
    assert wait_until(lambda: len(g.record.moves) >= 6, 10_000)
    first = [n.move.uci() for n in g.record.moves[:4]]
    assert first[0] in ("e2e4", "d2d4") and all(n.comment == "book" for n in g.record.moves[:2])
    assert g.record.moves[4].comment == ""  # the engines' own moves after that
    # Out of book earlier when the line leaves it.
    assert all(n.eval is not None and n.eval.score is None for n in g.record.moves[:2])
    g.abort()
    g.shutdown()


# --- Panels and window -----------------------------------------------------------------------------


def test_panels(qapp, tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(GAMES, encoding="utf-8")
    db = GameDatabase()
    db.add_file(str(path))
    panel = DatabasePanel()
    searches = Recorder(panel.search_changed)
    panel.show_games(db, db.search())
    assert panel.model.rowCount() == 4 and panel.model.data(panel.model.index(0, 1)) == "Ann"
    assert "4 of 4 games" in panel.status.text() and not panel.here.isEnabled()
    panel.model.sort(1, Qt.SortOrder.DescendingOrder)
    assert [panel.model.data(panel.model.index(r, 1)) for r in range(4)] == ["Dee", "Cy", "Bob", "Ann"]
    panel.text.setText("x")
    panel.result.setCurrentText("1-0")
    assert len(searches.calls) == 2 and panel.filters() == ("x", "1-0", False)
    opened = Recorder(panel.game_requested)
    panel._double_clicked(panel.model.index(0, 0))
    assert opened.calls == [(4,)]  # Dee's game, first after the sort
    explorer = ExplorerPanel()
    explorer.show_position(db, Position.start(), 0)
    assert "indexing" in explorer.title.text() or "open a database" in explorer.title.text()
    db.build_index(workers=0, wait=True)
    explorer.show_position(db, Position.start(), 0)
    assert explorer.table.rowCount() == 2 and explorer.table.item(0, 0).text() == "e4"
    assert explorer.table.item(0, 1).text() == "2" and explorer.table.item(0, 2).data(Qt.ItemDataRole.UserRole) == (1, 1, 0)
    assert "3 games reached" in explorer.title.text()
    chosen = Recorder(explorer.move_selected)
    explorer.table.cellClicked.emit(1, 0)
    assert chosen.calls == [(Move.from_uci("d2d4"),)]
    explorer.show_position(db, start_after("a2a3"), 1)
    assert explorer.table.rowCount() == 0 and "no game reached" in explorer.title.text()
    explorer.figurines = True
    explorer.show_position(db, start_after("e2e4", "e7e5"), 2)
    assert explorer.table.item(0, 0).text() == "♘f3"
    db.build_index(plies=4, workers=0, wait=True)  # 4 plies: 2 moves
    panel.show_games(db, db.search())
    assert "positions indexed for the first 2 moves" in panel.status.text()
    explorer.show_position(db, start_after("e2e4", "e7e5", "g1f3", "b8c6"), 4)
    assert explorer.table.rowCount() == 0 and "only the first 2 moves of each game are indexed" in explorer.title.text()


def test_database_in_the_window(qapp, tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(GAMES, encoding="utf-8")
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    settings.setValue("profiles", profiles_to_json([fake(name="bot")]))
    w = MainWindow(settings)
    w.resize(1100, 750)
    w.show()
    QApplication.processEvents()
    try:
        assert w.open_database_from(str(path), wait=True)
        assert w.tabs.currentWidget() is w.database_panel and w.database_panel.model.rowCount() == 4
        assert w.status.text() == "games.pgn: positions of 4 games indexed"
        assert w.explorer_panel.table.rowCount() == 2  # the start position's moves
        w.open_database_game(1)
        assert w.game.record.tags["White"] == "Ann" and len(w.game.record.moves) == 6 and w.tabs.currentIndex() == 0
        w.go_to(2)  # after 1. e4 e5: the explorer follows the board
        assert [m.uci() for m in w.explorer_panel.moves] == ["g1f3"]
        w.explorer_panel.table.cellClicked.emit(0, 0)  # the game's own move: shown
        assert w.view_node is w.game.record.moves[2]
        w.go_to(0)
        w.explorer_panel.table.cellClicked.emit(1, 0)  # 1. d4: a variation
        assert w.view_node.san == "d4" and w.view_node.starts_variation()
        w.database_panel.here.setChecked(True)  # games reaching the board's position (after 1. d4)
        assert [w.database_panel.model.data(w.database_panel.model.index(r, 0))
                for r in range(w.database_panel.model.rowCount())] == ["3"]
        w.database_panel.here.setChecked(False)
        # A game against an engine with the book: its first moves come from the games.
        w.start_game(None, w.profiles[0], TimeControl(movetime_ms=20), None, 4)
        assert w.game.book is w.database.index and w.game.book_plies == 4
        w.game.abort()
        w.database.clear()
        assert w.database_panel.model.rowCount() == 0 and w.explorer_panel.table.rowCount() == 0
    finally:
        w.close()


def test_index_build_falls_back_and_fails_cleanly(qapp, tmp_path, monkeypatch):
    import concurrent.futures

    import chessgpu_gui.database as database

    path = tmp_path / "games.pgn"
    path.write_text(GAMES, encoding="utf-8")
    db = GameDatabase()
    db.add_file(str(path))

    class NoWorkers:  # worker processes that can't start
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise concurrent.futures.BrokenExecutor("no workers")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", NoWorkers)
    messages, ready = Recorder(db.index_message), Recorder(db.index_ready)
    db.build_index(workers=2, chunk=3, wait=True)
    assert messages.calls and "indexing in this process instead" in messages.calls[0][0]
    assert db.index is not None and db.index.indexed == 4 and ready.calls and not db.building
    # A bug while indexing: reported, never "indexing..." forever.
    monkeypatch.setattr(database, "index_games", lambda *args: 1 / 0)
    failed = Recorder(db.index_failed)
    db.add_file(str(path))
    db.build_index(workers=0, wait=True)
    assert failed.calls and "ZeroDivisionError" in failed.calls[0][0] and not db.building and db.index is None
