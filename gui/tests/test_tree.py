"""The game tree: variations, comments, NAGs and commands in PGN, and the
record's tree operations."""

import pathlib

import pytest
from PySide6.QtGui import QColor

from chessgpu_gui.game import (GameController, GameRecord, Node, TimeControl, format_clock_command,
                               parse_clock_command, parse_shapes, shapes_commands)
from chessgpu_gui.move_tree import MoveTreeView, line_text, nag_text, toggle_nag
from chessgpu_gui.pgn import PgnError, PgnGame, join_commands, read_pgn, split_commands, write_pgn
from chessgpu_gui.rules import START_FEN, IllegalMoveError, Move, Position, parse_square
from chessgpu_gui.score import MoveEval, PositionEval, Score, parse_eval_command, parse_eval_comment
from chessgpu_gui.theme import current_theme
from qt_helpers import Recorder


def sq(name):
    return parse_square(name)


def dump(node, pos=None, depth=0):
    """A PgnNode tree as nested lists of (san, nags, comment, starting comment)."""
    pos = pos or Position.start()
    out = []
    for c in node.children:
        out.append((pos.san(c.move), c.nags, c.comment, c.starting_comment, dump(c, pos.push(c.move))))
    return out


def record_dump(node):
    return [(c.san, c.nags, c.comment, dict(c.arrows), dict(c.circles), c.clock_ms, c.review, c.eval, c.note,
             record_dump(c)) for c in node.children]


def sans(nodes):
    return [n.san for n in nodes]


# --- PGN trees ---------------------------------------------------------------------------------


def test_reads_variations_comments_and_nags():
    g = read_pgn("""[Event "x"]
{Intro} 1.e4 e5 2. Nf3!? {Main line} (2. f4 exf4 (2... d5) 3. Nf3) 2... Nc6 $1 3.Bb5 a6 ({Also} 3... Nf6) *""")[0]
    assert g.root.comment == "Intro"
    assert [Position.start().san(m) for m in g.moves[:1]] == ["e4"] and len(g.moves) == 6
    e5 = g.root.children[0].children[0]
    nf3, f4 = e5.children
    assert nf3.nags == [5] and nf3.comment == "Main line"
    assert len(f4.children) == 2 and len(f4.children[0].children) == 1  # exf4 (then Nf3), or d5
    assert nf3.children[0].nags == [1]
    a6, nf6 = nf3.children[0].children[0].children
    assert nf6.starting_comment == "Also" and a6.starting_comment == ""


def test_writes_variations():
    g = read_pgn("1. e4 e5 (1... c5 2. Nf3 (2. c3 d5) d6) 2. Nf3 $1 {good} Nc6 *")[0]
    text = write_pgn(g)
    assert text.split("\n\n")[1].strip() == \
        "1. e4 e5 (1... c5 2. Nf3 (2. c3 d5) 2... d6) 2. Nf3 $1 {good} 2... Nc6 *"
    again = read_pgn(text)[0]
    assert dump(again.root) == dump(g.root)
    first = read_pgn("{Intro} 1. e4 ({Also} 1. d4) e5 *")[0]
    assert write_pgn(first).split("\n\n")[1].strip() == "{Intro} 1. e4 ({Also} 1. d4) 1... e5 *"


def test_variation_round_trip_from_a_fen_and_wrapping():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    g = PgnGame(start=Position.from_fen(fen))
    node, pos = g.root, g.start
    for san in ["c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]:
        move = pos.parse_san(san)
        node = node.add(move, ("a long comment that keeps going on and on " * (san == "d4")).strip())
        pos = pos.push(move)
    branch = g.root.children[0]  # 1... c5
    alt = branch.add(Position.from_fen(fen).push(branch.move).parse_san("Nc3"))
    alt.nags = [6, 14]
    text = write_pgn(g)
    assert all(len(line) <= 80 for line in text.splitlines())
    back = read_pgn(text)[0]
    assert dump(back.root, back.start) == dump(g.root, g.start)
    assert back.mainline()[3].comment == "a long comment that keeps going on and on"


def test_tolerates_broken_variations():
    g = read_pgn("1. e4 e5 (1... c5 2. Ke3 Nc6 (2... d6)) 2. Nf3 *")[0]
    assert len(g.moves) == 3
    c5 = g.root.children[0].children[1]
    assert c5.children == []  # dropped from the illegal move on, its sub-variation too
    g = read_pgn("1. e4 (1. -- e5) e5 *")[0]
    assert len(g.root.children) == 1 and len(g.moves) == 2
    g = read_pgn("( 1. d4 ) 1. e4 *")[0]
    assert len(g.root.children) == 1 and len(g.moves) == 1
    g = read_pgn("1. e4 (1. d4 (1. c4) (( junk )) d5) e5 *")[0]
    assert len(g.root.children) == 3 and len(g.moves) == 2
    d4 = g.root.children[1]
    assert [Position.start().push(d4.move).san(c.move) for c in d4.children] == ["d5"]
    with pytest.raises(PgnError, match="move 2"):
        read_pgn("1. e4 e5 2. Ke3 (2. Nf3) *")
    with pytest.raises(PgnError):
        read_pgn("1. e4 -- *")
    with pytest.raises(PgnError):
        read_pgn("1. e4 (1. d4)) *")


def test_comment_before_the_tags_belongs_to_no_game():
    games = read_pgn('{ exported by someone }\n[Event "a"]\n1. e4 *\n{ before }\n[Event "b"]\n1. d4 *')
    assert [g.root.comment for g in games] == ["", ""] and len(games) == 2
    games = read_pgn('[Event "a"]\n{ game comment } 1. e4 *')
    assert games[0].root.comment == "game comment"


def test_annotation_tokens():
    g = read_pgn("1. e4 ! e5 ?? 2. Nf3 !? ?! Nc6 !!? *")[0]
    assert [n.nags for n in g.mainline()] == [[1], [4], [5, 6], []]


def test_commands():
    text, commands = split_commands("[%clk 0:01:02] nice [%cal Ge2e4,Rd7d5]  move [%csl Yd4] [%cal Bb1c3]")
    assert text == "nice move"
    assert commands == {"clk": "0:01:02", "cal": "Ge2e4,Rd7d5,Bb1c3", "csl": "Yd4"}
    assert join_commands("nice move", {"clk": "0:01:02", "cal": ""}) == "nice move [%clk 0:01:02]"
    assert join_commands("", {"csl": "Gd4"}) == "[%csl Gd4]"
    assert split_commands("no commands")[1] == {}
    arrows, circles = parse_shapes("Ge2e4,Rd7d5,Xa1a2,Gh9h1,Ba1", "Yd4,Gc3c3")
    assert arrows == {(sq("e2"), sq("e4")): "green", (sq("d7"), sq("d5")): "red"}
    assert circles == {sq("a1"): "blue", sq("d4"): "yellow", sq("c3"): "green"}
    assert shapes_commands(arrows, {sq("d4"): "yellow"}) == {"cal": "Ge2e4,Rd7d5", "csl": "Yd4"}
    for ms, text in ((0, "0:00:00"), (61_000, "0:01:01"), (3_723_400, "1:02:03.4"), (9_950, "0:00:10")):
        assert format_clock_command(ms) == text
    assert parse_clock_command("1:02:03.4") == pytest.approx(3_723_400)
    assert parse_clock_command("0:00:59") == 59_000 and parse_clock_command("soon") is None


def test_eval_commands_and_times():
    for e, text in ((PositionEval(Score(cp=35), 18), "0.35,18"), (PositionEval(Score(cp=-120)), "-1.20"),
                    (PositionEval(Score(mate=3), 30), "#3,30"), (PositionEval(Score(mate=-2)), "#-2"),
                    (PositionEval(Score(mate=0, given=True)), "#0"), (PositionEval(Score(mate=0)), "#-0")):
        assert e.command() == text
        assert parse_eval_command(text) == e
    assert parse_eval_command("+0.5") == PositionEval(Score(cp=50))
    assert parse_eval_command("good") is None and parse_eval_command("#") is None
    assert parse_eval_comment("3.500s") == MoveEval(None, None, 3.5)
    assert parse_eval_comment("3.5 s") is None


# --- Record trees --------------------------------------------------------------------------------


def record_of(*sans, fen=START_FEN):
    r = GameRecord(Position.from_fen(fen))
    for s in sans:
        r.push(r.position.parse_san(s))
    return r


def test_tree_operations():
    r = record_of("e4", "e5", "Nf3", "Nc6")
    e4, e5 = r.moves[0], r.moves[1]
    c5 = r.add(e4, e4.position.parse_san("c5"))
    assert c5.starts_variation() and not c5.is_main() and e5.is_main()
    assert r.add(e4, e4.position.parse_san("c5")) is c5 and len(e4.children) == 2  # an existing move: the same node
    assert r.add(e4, e4.position.parse_san("e5")) is e5
    end = r.add_line(c5, [c5.position.parse_san("Nf3"), Move.from_uci("d7d6")])
    assert sans(end.path()) == ["", "e4", "c5", "Nf3", "d6"] and end.uci_moves() == ["e2e4", "c7c5", "g1f3", "d7d6"]
    assert end.ply == 4 and c5.end() is end and sans(c5.line()) == ["", "e4", "c5", "Nf3", "d6"]
    with pytest.raises(IllegalMoveError):
        r.add(end, Move.from_uci("e1e8"))
    tail = r.add(r.end, r.position.parse_san("Bb5"))  # at the main line's end: it continues
    assert r.moves[-1] is tail and len(r.moves) == 5 and r.position == tail.position
    assert sans(r.nodes()) == ["", "e4", "e5", "Nf3", "Nc6", "Bb5", "c5", "Nf3", "d6"]

    r.promote(end)  # c5 trades places with e5
    assert e4.children == [c5, e5] and sans(r.moves) == ["e4", "c5", "Nf3", "d6"]
    r.promote(end)  # already the main line: nothing to do
    assert sans(r.moves) == ["e4", "c5", "Nf3", "d6"]
    r.make_main(tail)
    assert sans(r.moves) == ["e4", "e5", "Nf3", "Nc6", "Bb5"] and e4.children == [e5, c5]
    assert r.contains(end) and r.contains(r.root)
    assert r.delete(e5) is e4
    assert not r.contains(e5) and not r.contains(tail) and sans(r.moves) == ["e4", "c5", "Nf3", "d6"]
    with pytest.raises(ValueError):
        r.delete(r.root)
    assert not r.contains(Node(Position.start()))


def test_deep_variations():
    r = record_of("e4", "e5", "Nf3")
    a = r.add(r.moves[0], Move.from_uci("c7c5"))
    b = r.add(a, Move.from_uci("g1f3"))
    c = r.add(a, Move.from_uci("b1c3"))
    d = r.add(c, Move.from_uci("b8c6"))
    r.make_main(d)
    assert sans(r.moves) == ["e4", "c5", "Nc3", "Nc6"] and a.children == [c, b]
    r.promote(r.moves[2].parent.children[1])  # Nf3 back above Nc3
    assert sans(r.moves) == ["e4", "c5", "Nf3"]
    r.promote(c)  # Nc3 (with Nc6) above Nf3 again: c is where it branches
    assert sans(r.moves) == ["e4", "c5", "Nc3", "Nc6"]


def test_pop_and_truncate_drop_variations():
    r = record_of("e4", "e5", "Nf3")
    r.add(r.moves[1], Move.from_uci("b1c3"))  # a variation of the last move
    r.pop()
    assert sans(r.moves) == ["e4", "e5"] and r.moves[1].children == []
    r.add(r.root, Move.from_uci("d2d4"))
    r.truncate(0)
    assert r.moves == [] and r.root.children == [] and r.positions == [r.start]


def test_repetitions_in_a_variation():
    r = record_of("Nf3", "Nf6", "e3")
    node = r.moves[1]
    for uci in ("f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"):
        node = r.add(node, Move.from_uci(uci))
    assert not node.is_main() and len(r.moves) == 3 and node.position.key() == r.start.key()
    assert GameRecord.repetitions_at(node) == 3 and GameRecord.repetitions_at(node.parent) == 2
    assert r.ending_at(node) == ("1/2-1/2", "Draw by threefold repetition")
    assert r.ending_at(r.moves[1]) is None
    mate = record_of("f3", "e5", "g4")
    qh4 = mate.add(mate.end, Move.from_uci("d8h4"))
    assert mate.ending() == ("0-1", "Black mates") and mate.ending_at(qh4) == ("0-1", "Black mates")


def test_record_tree_pgn_round_trip():
    r = record_of("e4", "e5", "Nf3", "Nc6")
    r.root.comment = "A game"
    r.root.set_shapes({(sq("e2"), sq("e4")): "green"}, {})
    e4, e5, nf3, nc6 = r.moves
    e4.eval = MoveEval(Score(cp=30), 12, 1.5)
    e4.clock_ms = 59_000
    e5.nags = [2]
    e5.comment = "Better, by far, is c5"
    e5.set_shapes({(sq("c7"), sq("c5")): "blue"}, {sq("e5"): "red", sq("d4"): "yellow"})
    nf3.note = "tl=1.2s, nps=5000"
    nf3.review = PositionEval(Score(cp=40), 20)
    c5 = r.add(e4, Move.from_uci("c7c5"))
    c5.comment, c5.starting_comment, c5.nags = "Sicilian", "Instead", [5]
    r.add_line(c5, [Move.from_uci("g1f3"), Move.from_uci("d7d6")])
    r.result, r.reason = "0-1", "White loses on time, but Black can still mate"
    text = write_pgn(r.to_pgn())
    assert "[%cal Ge2e4]" in text and "[%clk 0:00:59]" in text and "[%eval 0.40,20]" in text
    back = GameRecord.from_pgn(read_pgn(text)[0])
    assert record_dump(back.root) == record_dump(r.root)
    assert back.root.comment == "A game" and back.root.arrows == r.root.arrows
    assert (back.result, back.reason) == ("0-1", "White loses on time, but Black can still mate")
    assert sans(back.moves) == ["e4", "e5", "Nf3", "Nc6"]


def test_reasons_and_notes_from_comments():
    g = read_pgn('1. e4 {+0.10/5 0.1s, tl=0.9s, nps=12} e5 {-0.20/6 0.2s, tl=0.8s, Black wins by adjudication} 0-1')[0]
    r = GameRecord.from_pgn(g)
    assert r.reason == "Black wins by adjudication"
    assert r.moves[0].note == "tl=0.9s, nps=12" and r.moves[0].comment == ""
    assert r.moves[1].eval == MoveEval(Score(cp=-20), 6, 0.2)
    g = read_pgn('1. e4 e5 {White is better, Black should defend} 1-0')[0]
    r = GameRecord.from_pgn(g)
    assert r.reason == "" and r.moves[1].comment == "White is better, Black should defend"
    g = read_pgn('1. e4 e5 {3.500s, nice, White resigns} 0-1')[0]
    r = GameRecord.from_pgn(g)
    assert (r.reason, r.moves[1].comment, r.moves[1].eval) == ("White resigns", "nice", MoveEval(None, None, 3.5))
    g = read_pgn('1. e4 e5 {Draw by threefold repetition} *')[0]  # an unfinished game has no reason
    assert GameRecord.from_pgn(g).reason == ""


# --- Exploring --------------------------------------------------------------------------------------


def test_explore_builds_variations(qapp):
    g = GameController()
    g.load(record_of("e4", "e5", "Nf3"))
    g.record.result = "1-0"
    changed = []
    g.changed.connect(lambda: changed.append(1))
    e4 = g.record.moves[0]
    c5 = g.explore(e4, Move.from_uci("c7c5"))
    assert c5 is not None and c5.parent is e4 and sans(g.record.moves) == ["e4", "e5", "Nf3"]
    assert g.record.result == "1-0" and changed == [1]
    assert g.explore(e4, Move.from_uci("e7e5")) is g.record.moves[1]  # the game's move
    end = g.explore(g.record.end, Move.from_uci("b8c6"))
    assert g.record.moves[-1] is end and len(g.record.moves) == 4
    assert g.explore(e4, Move.from_uci("e7e4")) is None  # illegal
    assert g.explore(Node(Position.start()), Move.from_uci("e2e4")) is None  # not in this game
    assert g.edit(g.record.promote, c5) is None and sans(g.record.moves) == ["e4", "c5"]
    assert g.edit(g.record.delete, c5) is e4 and sans(g.record.moves) == ["e4", "e5", "Nf3", "Nc6"]


def test_no_exploring_during_a_game(qapp):
    g = GameController()
    g.new_game(None, None, TimeControl())
    assert g.status == g.RUNNING
    assert g.explore(g.record.root, Move.from_uci("e2e4")) is None and g.record.moves == []
    assert g.edit(g.record.add, g.record.root, Move.from_uci("e2e4")) is None and g.record.moves == []
    g.abort()


# --- The move tree view --------------------------------------------------------------------------

DATA = pathlib.Path(__file__).resolve().parent / "data"


def annotated():
    return GameRecord.from_pgn(read_pgn((DATA / "annotated.pgn").read_text(encoding="utf-8"))[0])


def fragment_of(view, node):
    i = next(k for k, n in enumerate(view._nodes) if n is node)
    block = view.document().begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            frag = it.fragment()
            if frag.isValid() and frag.charFormat().anchorHref() == f"n:{i}":
                return frag
            it += 1
        block = block.next()
    return None


def test_move_tree_layout(qapp):
    r = annotated()
    view = MoveTreeView()
    view.resize(360, 500)
    view.set_tree(r, r.moves[2])
    texts = [t for _, t in view.anchors()]
    assert texts[:14] == ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Nf6", "Ng5", "d4", "exd4", "O-O", "d5", "exd5",
                          "Na5"]  # a variation right after the move it replaces, its own variation inside
    assert "Nc3?!" in texts and "d5!?" in texts and "Qe2??" in texts and "Nxh7!" in texts
    plain = view.toPlainText()
    assert "An open game where White goes wrong early." in plain and "Giving up the e4 pawn." in plain
    assert "(4. d4 exd4 5. O-O)" in plain and "7. ..." not in plain and "−+" in plain
    assert plain.rstrip().endswith("0-1")
    shown_nodes = [n for n, _ in view.anchors()]
    assert len(shown_nodes) == len(r.nodes()) - 1 and {id(n) for n in shown_nodes} == {id(n) for n in r.nodes()[1:]}
    shown = fragment_of(view, r.moves[2])
    assert shown.charFormat().background().color() == QColor(current_theme().accent)
    other = fragment_of(view, r.moves[3])
    assert other.charFormat().background().color() != QColor(current_theme().accent)
    variation_move = r.moves[5].parent.children[1]  # 3... Nf6
    view.set_tree(r, variation_move)
    assert fragment_of(view, variation_move).charFormat().background().color() == QColor(current_theme().accent)
    assert view.current is variation_move


def test_move_tree_menu_and_tooltip(qapp):
    r = annotated()
    view = MoveTreeView()
    asked = Recorder(view.action_requested)
    main_move, variation = r.moves[12], r.moves[12].parent.children[1]  # 7. Nc3, 7. Bd2
    view.set_tree(r, main_move)
    menu = view._menu(variation)
    actions = {a.text(): a for a in menu.actions()}
    annotate = {a.text().split("\t")[0]: a for a in actions["Annotate"].menu().actions() if a.text()}
    assert annotate["?!"].isChecked() is False and annotate["!"].isCheckable()
    assert actions["Promote variation"].isEnabled() and actions["Make main line"].isEnabled()
    for text in ("Promote variation", "Make main line", "Delete from here", "Edit comment", "Copy line", "Copy FEN"):
        actions[text].trigger()
    annotate["!!"].trigger()
    annotate["±"].trigger()
    names = [c[0] for c in asked.calls]
    assert names == ["promote", "make_main", "delete", "comment", "copy_line", "copy_fen", "nag:3", "nag:16"]
    assert all(c[1] is variation for c in asked.calls)
    menu = view._menu(main_move)
    actions = {a.text(): a for a in menu.actions()}
    assert not actions["Promote variation"].isEnabled() and not actions["Make main line"].isEnabled()
    view.can_edit = False
    actions = {a.text(): a for a in view._menu(variation).actions()}
    assert not actions["Delete from here"].isEnabled() and not actions["Promote variation"].isEnabled()
    assert MoveTreeView.tooltip(main_move) == "Dubious move"
    main_move.clock_ms = 65_000
    main_move.note = "tl=1s"
    assert MoveTreeView.tooltip(main_move) == "Dubious move\nclock 1:05\ntl=1s"


def test_nag_helpers():
    assert toggle_nag([], 1) == [1] and toggle_nag([1], 1) == []
    assert toggle_nag([1, 16], 4) == [16, 4] and toggle_nag([4, 16], 18) == [4, 18]
    assert toggle_nag([146], 2) == [146, 2]
    assert nag_text([6, 16, 146, 250]) == ("?!", "± N $250")
    r = annotated()
    assert line_text(r.moves[3]) == "1. e4 e5 2. Nf3 Nc6"
    variation = r.moves[5].parent.children[1]
    assert line_text(variation) == "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6"
    assert line_text(variation, figurines=True).endswith("3. ♗c4 ♘f6")
    black_first = record_of("e5", "Nf3", "Nc6", fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert line_text(black_first.end) == "1... e5 2. Nf3 Nc6"


def test_null_move():
    start = Position.start()
    after = start.null_move()
    assert after.turn != start.turn and after.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 1"
    e4 = start.push(Move.from_uci("e2e4"))  # an en passant square goes with a pass
    assert e4.null_move().fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
    with pytest.raises(IllegalMoveError):
        Position.from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").null_move()  # no passing in check
