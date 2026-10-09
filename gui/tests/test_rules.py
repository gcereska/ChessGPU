"""The GUI's chess rules: perft against the suites (counts verified with
Stockfish), FEN, SAN, every way the rules end a game, and a cross-check of
move lists and positions against the engine."""

import random

import pytest

from chessgpu_gui.rules import (BLACK, START_FEN, WHITE, Ending, FenError, IllegalMoveError, Move, Position,
                                parse_square, perft, rules_ending, square_name)
from helpers import SimpleEngine, all_perft_positions

FAST_NODES = 250_000  # perft depths up to this many nodes run by default
SLOW_NODES = 5_000_000  # and these with -m slow

POSITIONS = all_perft_positions()


def play(fen, *moves):
    """The position after `moves` (SAN or UCI) from `fen`."""
    pos = Position.from_fen(fen)
    for text in moves:
        pos = pos.push(pos.parse_san(text))
    return pos


# --- Perft -------------------------------------------------------------------


@pytest.mark.parametrize("fen,counts", POSITIONS, ids=[f"pos{i}" for i in range(len(POSITIONS))])
def test_perft(fen, counts):
    pos = Position.from_fen(fen)
    checked = 0
    for depth in sorted(counts):
        if counts[depth] > FAST_NODES:
            break
        assert perft(pos, depth) == counts[depth], f"{fen} depth {depth}"
        checked += 1
    assert checked >= 2


@pytest.mark.slow
@pytest.mark.parametrize("fen,counts", POSITIONS, ids=[f"pos{i}" for i in range(len(POSITIONS))])
def test_perft_deep(fen, counts):
    pos = Position.from_fen(fen)
    for depth in sorted(counts):
        if FAST_NODES < counts[depth] <= SLOW_NODES:
            assert perft(pos, depth) == counts[depth], f"{fen} depth {depth}"


# --- FEN ---------------------------------------------------------------------


@pytest.mark.parametrize("fen", [fen for fen, _ in POSITIONS])
def test_fen_round_trip(fen):
    assert Position.from_fen(fen).fen() == fen


def test_fen_defaults_and_normalization():
    assert Position.from_fen("4k3/8/8/8/8/8/8/4K3 w - -").fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    # Castling letters come out in KQkq order.
    assert Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w qkQK - 0 1").castling == "KQkq"


@pytest.mark.parametrize("fen,message", [
    ("4k3/8/8/8/8/8/8/4K3 w -", "expected 4 to 6 fields"),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra", "expected 4 to 6 fields"),
    ("4k3/8/8/8/8/8/4K3 w - - 0 1", "8 ranks"),
    ("4k3/8/8/8/8/8/8/4K4 w - - 0 1", "more than 8 squares"),
    ("4k3/8/8/8/8/8/8/4K2 w - - 0 1", "does not have 8 squares"),
    ("4k3/8/8/8/8/8/8/4X3 w - - 0 1", "unknown piece letter"),
    ("4k3/8/8/8/8/8/8/4K3 x - - 0 1", "side to move"),
    ("4k3/8/8/8/8/8/8/4K3 w X - 0 1", "unknown castling letter"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KK - 0 1", "repeated"),
    ("4k3/8/8/8/8/8/8/4K3 w - z9 0 1", "invalid en passant square"),
    ("4k3/8/8/8/8/8/8/4K3 w - e3 0 1", "rank 6"),
    ("4k3/8/8/8/8/8/8/4K3 w - - x 1", "halfmove"),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 -1", "fullmove"),
    ("8/8/8/8/8/8/8/4K3 w - - 0 1", "exactly one king"),
    ("4k3/8/8/8/8/8/8/2K1K3 w - - 0 1", "exactly one king"),
    ("4k3/8/8/8/8/8/8/P3K3 w - - 0 1", "pawn on the first or last rank"),
    ("P3k3/8/8/8/8/8/8/4K3 w - - 0 1", "pawn on the first or last rank"),
    ("4k3/8/8/8/8/8/8/4K3 w K - 0 1", "castling right K"),
    ("4k3/8/8/8/8/8/8/R3K3 w KQ - 0 1", "castling right K"),
    ("4k3/8/8/8/8/8/8/4K2R w Q - 0 1", "castling right Q"),
    ("4k3/8/8/8/8/8/8/R3K2R w k - 0 1", "castling right k"),
    ("4k2r/8/8/8/8/8/8/R3K2R w q - 0 1", "castling right q"),
    ("r3k2r/8/8/8/8/8/8/R2K3R w K - 0 1", "castling right K"),
    ("4k3/8/8/8/8/8/8/4K3 w - e6 0 1", "double pawn push"),
    ("4k3/4p3/8/4p3/8/8/8/4K3 w - e6 0 1", "double pawn push"),
    ("4k3/8/8/8/4P3/8/8/4K3 w - e3 0 1", "rank 6"),
    ("4k3/8/8/8/8/8/8/4RK2 w - - 0 1", "not to move is in check"),
])
def test_fen_errors(fen, message):
    with pytest.raises(FenError, match=message):
        Position.from_fen(fen)


def test_squares_and_uci_moves():
    assert square_name(0) == "a1" and square_name(63) == "h8" and square_name(parse_square("e4")) == "e4"
    with pytest.raises(ValueError):
        parse_square("i9")
    assert Move.from_uci("e7e8q") == Move(52, 60, "q") and Move(52, 60, "q").uci() == "e7e8q"
    for bad in ("e2", "e2e9", "e7e8k", "e2e4qq"):
        with pytest.raises(ValueError):
            Move.from_uci(bad)


# --- Moves -------------------------------------------------------------------


def test_double_push_sets_en_passant_square_and_clocks():
    pos = play(START_FEN, "e4")
    assert pos.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    pos = pos.push(pos.parse_san("Nf6"))
    assert pos.fen() == "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
    pos = pos.push(pos.parse_san("e5"))
    assert pos.halfmove == 0 and pos.ep is None


def test_push_rejects_illegal_moves():
    pos = Position.start()
    with pytest.raises(IllegalMoveError):
        pos.push(Move.from_uci("e2e5"))
    pinned = Position.from_fen("4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1")
    assert not pinned.is_legal(Move.from_uci("e2d3"))  # the bishop is pinned


def test_en_passant_capture_removes_the_pawn():
    pos = play(START_FEN, "e4", "a6", "e5", "d5")
    assert pos.ep == parse_square("d6")
    ep = pos.parse_san("exd6")
    assert pos.is_en_passant(ep) and pos.is_capture(ep)
    after = pos.push(ep)
    assert after.board[parse_square("d5")] == "." and after.board[parse_square("d6")] == "P"
    assert after.halfmove == 0
    # One move later the right is gone.
    late = play(START_FEN, "e4", "a6", "e5", "d5", "a3", "a5")
    assert not any(late.is_en_passant(m) for m in late.legal_moves())


def test_en_passant_that_exposes_the_king_is_illegal():
    pos = Position.from_fen("8/8/8/KPp4r/8/8/8/6k1 w - c6 0 1")
    assert not pos.is_legal(Move.from_uci("b5c6"))
    assert not pos.has_legal_en_passant()


def test_castling_rules():
    pos = Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"O-O", "O-O-O"} <= {pos.san(m) for m in pos.legal_moves()}
    after = pos.push(pos.parse_san("O-O"))
    assert after.board[parse_square("g1")] == "K" and after.board[parse_square("f1")] == "R"
    assert after.castling == "kq"
    after = pos.push(pos.parse_san("O-O-O"))
    assert after.board[parse_square("c1")] == "K" and after.board[parse_square("d1")] == "R"
    # A rook move loses that side's right; capturing a rook loses the opponent's.
    assert pos.push(pos.parse_san("Rb1")).castling == "Kkq"
    assert pos.push(pos.parse_san("Rxa8+")).castling == "Kk"
    assert pos.push(pos.parse_san("Kd1")).castling == "kq"
    # Not out of check (e1), through check (f1, d1) or into check (g1, c1).
    out_of_check = sans("4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert not {"O-O", "O-O-O"} & out_of_check
    assert "O-O" not in sans("4k3/5r2/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert "O-O" not in sans("4k3/6r1/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert "O-O-O" not in sans("4k3/3r4/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert "O-O-O" not in sans("4k3/2r5/8/8/8/8/8/R3K2R w KQ - 0 1")
    # The rook may pass an attacked square (b1); only the king's path counts.
    assert {"O-O", "O-O-O"} <= sans("4k3/1r6/8/8/8/8/8/R3K2R w KQ - 0 1")
    # Pieces in between block castling.
    assert "O-O-O" not in sans("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1")
    assert "O-O" not in sans("4k3/8/8/8/8/8/8/R3K1NR w KQ - 0 1")


def sans(fen):
    pos = Position.from_fen(fen)
    return {pos.san(m) for m in pos.legal_moves()}


def test_promotions():
    pos = Position.from_fen("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    names = sans("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert {"e8=Q+", "e8=R+", "e8=B", "e8=N", "exd8=Q+", "exd8=R+", "exd8=B", "exd8=N"} <= names
    after = pos.push(pos.parse_san("exd8=N"))
    assert after.board[parse_square("d8")] == "N"
    black = Position.from_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1").push(Move.from_uci("a2a1r"))
    assert black.board[parse_square("a1")] == "r"
    with pytest.raises(IllegalMoveError):
        pos.parse_san("e8")  # a promotion needs its piece


# --- SAN ---------------------------------------------------------------------


@pytest.mark.parametrize("fen,uci,san", [
    (START_FEN, "g1f3", "Nf3"),
    (START_FEN, "e2e4", "e4"),
    ("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", "b1d2", "Nbd2"),
    ("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", "f3d2", "Nfd2"),
    ("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "a1a3", "R1a3"),
    ("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "a5a3", "R5a3"),
    ("7k/8/8/8/8/3Q4/8/3QKQ2 w - - 0 1", "d1e2", "Qd1e2"),
    ("7k/8/8/8/8/3Q4/8/3QKQ2 w - - 0 1", "d3e2", "Q3e2"),
    ("7k/8/8/8/8/3Q4/8/3QKQ2 w - - 0 1", "f1e2", "Qfe2"),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5", "exd5"),
    ("rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", "exd6"),
    ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2", "d8h4", "Qh4#"),
    ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", "a1a8", "Ra8+"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "O-O"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "O-O-O"),
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", "O-O"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", "Ra8#"),
    ("4k3/8/8/8/8/8/1N3N2/4K3 w - - 0 1", "b2d1", "Nbd1"),
    ("4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1", "a1b3", "Nab3"),
])
def test_san(fen, uci, san):
    pos = Position.from_fen(fen)
    move = pos.parse_uci(uci)
    assert pos.san(move) == san
    assert pos.parse_san(san) == move


def walk(fen, plies, rng):
    """Positions along a random game from `fen`."""
    pos = Position.from_fen(fen)
    out = [pos]
    for _ in range(plies):
        moves = pos.legal_moves()
        if not moves:
            break
        pos = pos.push(rng.choice(moves))
        out.append(pos)
    return out


def test_san_round_trip_everywhere():
    rng = random.Random(7)
    checked = 0
    for fen, _ in POSITIONS:
        for pos in walk(fen, 40, rng):
            names = set()
            for m in pos.legal_moves():
                san = pos.san(m)
                assert san not in names, f"two moves read {san} in {pos.fen()}"
                names.add(san)
                assert pos.parse_san(san) == m
                assert pos.parse_san(m.uci()) == m
                assert pos.parse_san(san.rstrip("+#") + "!?") == m
                checked += 1
    assert checked > 10_000


def test_parse_san_variants_and_errors():
    pos = Position.start()
    nf3 = Move.from_uci("g1f3")
    for text in ("Nf3", "Ng1f3", "Ng1-f3", "Nf3+", "Nf3!", " Nf3 ", "g1f3"):
        assert pos.parse_san(text) == nf3
    assert play(START_FEN, "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5").parse_san("0-0") == Move.from_uci("e1g1")
    for bad in ("Nf4", "e5", "zz", "", "O-O", "Ke2", "Qxd7"):
        with pytest.raises(IllegalMoveError):
            pos.parse_san(bad)
    two_knights = Position.from_fen("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
    with pytest.raises(IllegalMoveError, match="ambiguous"):
        two_knights.parse_san("Nd2")
    with pytest.raises(IllegalMoveError):
        pos.parse_uci("e2e5")


# --- How games end -------------------------------------------------------------


def test_checkmate_and_stalemate():
    mate = play(START_FEN, "f3", "e5", "g4", "Qh4#")
    assert mate.is_checkmate() and rules_ending(mate, 1) == Ending.CHECKMATE
    stale = Position.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.is_stalemate() and rules_ending(stale, 1) == Ending.STALEMATE
    assert rules_ending(Position.start(), 1) is None


@pytest.mark.parametrize("fen,dead", [
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),
    ("4k3/8/8/8/8/8/8/4KB2 w - - 0 1", True),
    ("4k3/8/8/8/8/8/8/4KN2 w - - 0 1", True),
    ("4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1", True),  # bishops on the same color (c1, f8: dark)
    ("4k1b1/8/8/8/8/8/8/2B1K3 w - - 0 1", False),  # opposite colors: a helpmate exists
    ("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1", False),  # two bishops on both colors
    ("4k3/8/8/8/8/8/8/3NKN2 w - - 0 1", False),  # two knights can be mated with help
    ("4kn2/8/8/8/8/8/8/4KN2 w - - 0 1", False),
    ("4kn2/8/8/8/8/8/8/4KB2 w - - 0 1", False),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", False),
    ("4k3/8/8/8/8/8/8/4KR2 w - - 0 1", False),
    ("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", False),
    ("2b1kb2/8/8/8/8/8/8/2B1K3 w - - 0 1", False),  # c8 is light
    ("4k3/8/8/8/8/8/8/B1B1KB1B w - - 0 1", False),
    ("4k3/8/8/8/8/8/8/B1B1K3 w - - 0 1", True),  # a1, c1: both dark
])
def test_insufficient_material(fen, dead):
    pos = Position.from_fen(fen)
    assert pos.is_insufficient_material() == dead
    assert (rules_ending(pos, 1) == Ending.INSUFFICIENT_MATERIAL) == dead


def test_fifty_move_rule_and_mate_on_the_hundredth_halfmove():
    assert rules_ending(Position.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 80"), 1) is None
    assert rules_ending(Position.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 80"), 1) == Ending.FIFTY_MOVES
    # The move that reaches halfmove 100 mates: the mate counts.
    mate = Position.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
    mate = mate.push(mate.parse_san("Ra8#"))
    assert mate.halfmove == 100 and rules_ending(mate, 1) == Ending.CHECKMATE


def test_threefold_repetition_by_keys():
    pos = Position.start()
    seen = {pos.key(): 1}
    endings = []
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"] * 2:
        pos = pos.push(pos.parse_san(san))
        seen[pos.key()] = seen.get(pos.key(), 0) + 1
        endings.append(rules_ending(pos, seen[pos.key()]))
    assert endings[:-1] == [None] * 7
    assert endings[-1] == Ending.THREEFOLD_REPETITION


def test_key_counts_en_passant_only_when_capturable():
    # 1.e4: no black pawn can take, so the position equals one without the square.
    after_e4 = play(START_FEN, "e4")
    assert after_e4.ep is not None and after_e4.key()[3] is None
    capturable = play(START_FEN, "e4", "a6", "e5", "d5")
    assert capturable.key()[3] == parse_square("d6")
    pinned = Position.from_fen("8/8/8/KPp4r/8/8/8/6k1 w - c6 0 1")
    assert pinned.key()[3] is None


def test_sides_and_check():
    pos = play(START_FEN, "e4", "f5", "Qh5+")
    assert pos.turn == BLACK and pos.in_check()
    assert pos.king_square(WHITE) == parse_square("e1")
    assert not Position.start().in_check()


# --- Against the engine ----------------------------------------------------------


def test_matches_engine_move_lists_and_positions(engine):
    """Every position of shallow trees from the perft suites and of random
    games: the same legal moves, and the same FEN after each move list."""
    e = SimpleEngine(engine)
    try:
        checked = 0
        for fen, _ in POSITIONS:
            root = Position.from_fen(fen)
            for m1 in root.legal_moves():
                p1 = root.push(m1)
                e.set_position(fen, [m1.uci()])
                assert e.fen() == p1.fen()
                assert e.legal_moves() == {m.uci() for m in p1.legal_moves()}, f"{fen} moves {m1.uci()}"
                checked += 1
        rng = random.Random(0x7EA)
        for game in range(60):
            pos = Position.start()
            moves = []
            while True:
                legal = pos.legal_moves()
                e.set_position(START_FEN, moves)
                assert e.fen() == pos.fen(), f"game {game}, moves {' '.join(moves)}"
                assert e.legal_moves() == {m.uci() for m in legal}, f"game {game}, moves {' '.join(moves)}"
                checked += 1
                if rules_ending(pos, 1) or len(moves) >= 300:
                    break
                m = rng.choice(legal)
                moves.append(m.uci())
                pos = pos.push(m)
        assert checked > 5_000
    finally:
        e.close()
