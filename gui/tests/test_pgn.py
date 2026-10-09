"""PGN reading and writing, and the score comments in move annotations."""

import pathlib

import pytest

from chessgpu_gui.pgn import LINE_LENGTH, PgnError, PgnGame, read_pgn, write_pgn, write_pgns
from chessgpu_gui.rules import START_FEN, Move, Position
from chessgpu_gui.score import MoveEval, Score, parse_eval_comment
from helpers import ROOT

DATA = pathlib.Path(__file__).resolve().parent / "data"


def game_from_sans(fen, sans, comments=None, result="*", tags=None):
    pos = Position.from_fen(fen)
    moves = []
    for s in sans:
        m = pos.parse_san(s)
        moves.append(m)
        pos = pos.push(m)
    return PgnGame(tags=dict(tags or {}), start=Position.from_fen(fen), moves=moves,
                   comments=list(comments or [""] * len(moves)), result=result)


# --- Scores ----------------------------------------------------------------------


def test_score_text_and_perspective():
    assert Score(cp=35).text() == "+0.35"
    assert Score(cp=-120).text() == "-1.20"
    assert Score(cp=0).text() == "0.00"
    assert Score(mate=3).text() == "+M3" and Score(mate=-2).text() == "-M2"
    assert Score(mate=0).text() == "-M0" and not Score(mate=0).winning()
    given = Score(mate=0).negated()
    assert given.text() == "+M0" and given.winning() and given.bar_cp() == 1000
    assert given.negated() == Score(mate=0)
    assert Score(mate=3).negated() == Score(mate=-3) and Score(cp=50).negated() == Score(cp=-50)
    assert Score(cp=5000).bar_cp() == 1000 and Score(cp=-5000).bar_cp() == -1000
    assert Score(mate=-1).bar_cp() == -1000
    with pytest.raises(ValueError):
        Score()
    with pytest.raises(ValueError):
        Score(cp=1, mate=1)
    with pytest.raises(ValueError):
        Score(mate=2, given=True)


@pytest.mark.parametrize("comment,score,depth,seconds", [
    ("+0.53/5 0.066s, tl=1.974s, latency=0.001s, n=278490, nps=4229086", Score(cp=53), 5, 0.066),
    ("-M6/6 0.035s, tl=0.485s", Score(mate=-6), 6, 0.035),
    ("0.00/4 0.091s", Score(cp=0), 4, 0.091),
    ("+M1/8 0.026s, tl=0.508s, White mates", Score(mate=1), 8, 0.026),
    ("-7.99/7 0.219s, Black wins by adjudication", Score(cp=-799), 7, 0.219),
    ("+1.00/12", Score(cp=100), 12, None),
    ("+M0/1 0.001s", Score(mate=0, given=True), 1, 0.001),
])
def test_parse_eval_comment(comment, score, depth, seconds):
    e = parse_eval_comment(comment)
    assert e == MoveEval(score, depth, seconds)


def test_eval_comment_round_trip_and_non_scores():
    for e in (MoveEval(Score(cp=35), 12, 1.234), MoveEval(Score(mate=-4), 20, 0.5), MoveEval(Score(cp=-7), 3, None)):
        assert parse_eval_comment(e.comment()) == e
    assert MoveEval(Score(cp=35), 12, 1.234).comment() == "+0.35/12 1.234s"
    for text in ("book", "", "great move", "12 nodes"):
        assert parse_eval_comment(text) is None


# --- Writing -----------------------------------------------------------------------


def test_write_format():
    game = game_from_sans(START_FEN, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"],
                          comments=["", "+0.10/9 0.5s", "", "", "a } brace", ""], result="1/2-1/2",
                          tags={"White": 'Quote "me" \\ here', "Black": "b", "Termination": "normal"})
    text = write_pgn(game)
    lines = text.splitlines()
    assert [l.split()[0][1:] for l in lines[:8]] == ["Event", "Site", "Date", "Round", "White", "Black", "Result",
                                                     "Termination"]
    assert lines[4] == '[White "Quote \\"me\\" \\\\ here"]'
    assert lines[6] == '[Result "1/2-1/2"]'
    assert "SetUp" not in text and "FEN" not in text
    movetext = text.split("\n\n")[1]
    assert movetext.startswith("1. e4 e5 {+0.10/9 0.5s} 2. Nf3 Nc6 3. Bb5 {a ) brace} 3... a6 1/2-1/2")
    assert text.endswith("1/2-1/2\n")


def test_write_from_fen_black_to_move_and_wrapping():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    sans = ["Nf6", "Nc3", "Ng8", "Nb1"] * 15
    game = game_from_sans(fen, sans, result="*")
    text = write_pgn(game)
    assert '[SetUp "1"]' in text and f'[FEN "{fen}"]' in text
    movetext = text.split("\n\n")[1]
    assert movetext.startswith("1... Nf6 2. Nc3 Ng8 3. Nb1")
    assert all(len(line) <= LINE_LENGTH for line in movetext.splitlines())
    assert len(movetext.splitlines()) > 3
    back = read_pgn(text)[0]
    assert back.start.fen() == fen and back.moves == game.moves


def test_write_rejects_bad_games():
    with pytest.raises(PgnError):
        write_pgn(PgnGame(result="2-0"))
    with pytest.raises(Exception):
        write_pgn(PgnGame(moves=[Move.from_uci("e2e5")]))


# --- Reading -----------------------------------------------------------------------


def test_round_trip():
    game = game_from_sans(START_FEN, ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"],
                          comments=["opening", "", "+0.20/10 1.000s", "", "", "", "", "-0.15/11 2.5s"],
                          result="0-1", tags={"Event": "Test", "White": "w", "Black": "b"})
    back = read_pgn(write_pgn(game))
    assert len(back) == 1
    g = back[0]
    assert g.moves == game.moves and g.comments == game.comments and g.result == "0-1"
    assert g.tags["Event"] == "Test" and g.tags["White"] == "w"


def test_read_tolerates_real_world_syntax():
    text = """
% an escape line, ignored
[Event "Casual"]
[White "A"] [Black "B"]
[Result "1-0"]

1.e4 e5 2. Nf3!? {Main line} (2. f4 exf4 (2... d5) 3. Nf3) 2... Nc6 $1 3.Bb5 ; a rest-of-line comment
a6 4. Ba4 Nf6 5. 0-0 Be7 6. Re1+? b5 7. Bb3 d6 8. c3 O-O 1-0

[Event "Second"]
1. d4 Nf6 2. c4 *
1. e4 c5
"""
    games = read_pgn(text)
    assert len(games) == 3
    g = games[0]
    assert g.result == "1-0" and g.tags["Black"] == "B"
    assert [Position.start().san(g.moves[0])] == ["e4"]
    assert len(g.moves) == 16
    assert g.comments[2] == "Main line"
    assert g.comments[4] == "a rest-of-line comment"
    assert g.positions()[-1].fen() == "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9"
    assert games[1].tags == {"Event": "Second"} and len(games[1].moves) == 3 and games[1].result == "*"
    assert games[2].tags == {} and len(games[2].moves) == 2 and games[2].result == "*"


def test_result_inside_a_variation_doesnt_end_the_game():
    games = read_pgn("1. e4 e5 (1... f6 2. d4 g5 3. Qh5# 1-0) 2. Nf3 Nc6 *")
    assert len(games) == 1
    assert len(games[0].moves) == 4 and games[0].result == "*"


def test_read_errors():
    with pytest.raises(PgnError, match="move 2"):
        read_pgn("1. e4 e5 2. Ke3 *")
    with pytest.raises(PgnError, match="FEN"):
        read_pgn('[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n1. Kd2 *')
    with pytest.raises(PgnError):
        read_pgn("1. e4 ) e5 *")
    with pytest.raises(PgnError):
        read_pgn("1. e4 e5 & *")


def test_reads_fastchess_games():
    games = read_pgn((DATA / "fastchess_sample.pgn").read_text())
    assert [g.result for g in games] == ["0-1", "1-0", "0-1"]
    for g in games:
        assert len(g.moves) == int(g.tags["PlyCount"])
        assert g.start.fen() == g.tags["FEN"]
        evals = [parse_eval_comment(c) for c in g.comments]
        assert all(e is not None and e.depth >= 1 and e.seconds is not None for e in evals)
        again = read_pgn(write_pgn(g))[0]
        assert again.moves == g.moves and again.comments == g.comments and again.result == g.result
    assert games[0].tags["White"] == "thread" and games[0].tags["Termination"] == "adjudication"
    assert len(read_pgn(write_pgns(games))) == 3


def test_reads_every_match_game_if_present():
    """Every game of a local corpus of 690 match games (the smoke, eval and model
    matches; build output, skipped if absent)."""
    folder = ROOT / "build" / "matches" / "corpus"
    files = sorted(folder.glob("*.pgn")) if folder.exists() else []
    if not files:
        pytest.skip("no match PGNs in build/matches/corpus")
    total = 0
    for f in files:
        games = read_pgn(f.read_text())
        for g in games:
            assert len(g.moves) == int(g.tags["PlyCount"]), f"{f.name} round {g.tags.get('Round')}"
            assert g.result == g.tags["Result"]
        total += len(games)
    assert total == 690
