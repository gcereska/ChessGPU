"""Chess rules for the GUI: positions, legal moves, FEN, SAN and how games end.

Written for this project, independently of the engine (C++) and of
tests/reference_chess.h, so the three can check each other: the tests compare
this module with the perft suites (counts verified with Stockfish) and with
the engine's own move lists.

Squares are numbered a1 = 0, b1 = 1, ..., h8 = 63. A board is a string of 64
characters, a1 first: FEN letters for pieces ("PNBRQK" white, "pnbrqk" black)
and "." for empty squares. Positions are immutable: push() returns a new one.
FEN validation matches the engine's (src/host/position.cpp), so every
position this module accepts is one the engine accepts too.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

WHITE, BLACK = 0, 1
EMPTY = "."
FILES = "abcdefgh"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PROMOTIONS = "qrbn"


class FenError(ValueError):
    """A FEN string that doesn't describe a valid position."""


class IllegalMoveError(ValueError):
    """A move (or SAN/UCI text) that isn't legal in the position."""


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return FILES[sq & 7] + str((sq >> 3) + 1)


def parse_square(text: str) -> int:
    if len(text) != 2 or text[0] not in FILES or text[1] not in "12345678":
        raise ValueError(f"invalid square '{text}'")
    return square(FILES.index(text[0]), int(text[1]) - 1)


def color_of(piece: str) -> int:
    return WHITE if piece.isupper() else BLACK


def other(color: int) -> int:
    return color ^ 1


# Geometry, precomputed per square.
def _targets(sq, steps):
    f, r = file_of(sq), rank_of(sq)
    return tuple(square(f + df, r + dr) for df, dr in steps if 0 <= f + df < 8 and 0 <= r + dr < 8)


def _ray(sq, df, dr):
    out = []
    f, r = file_of(sq) + df, rank_of(sq) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        out.append(square(f, r))
        f, r = f + df, r + dr
    return tuple(out)


_KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_STEPS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_ROOK_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_BISHOP_DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
KNIGHT_TARGETS = tuple(_targets(s, _KNIGHT_STEPS) for s in range(64))
KING_TARGETS = tuple(_targets(s, _KING_STEPS) for s in range(64))
ROOK_RAYS = tuple(tuple(_ray(s, df, dr) for df, dr in _ROOK_DIRS) for s in range(64))
BISHOP_RAYS = tuple(tuple(_ray(s, df, dr) for df, dr in _BISHOP_DIRS) for s in range(64))

# Squares whose piece moving away, or being captured, removes castling rights.
_CASTLING_SQUARES = {4: "KQ", 7: "K", 0: "Q", 60: "kq", 63: "k", 56: "q"}
_SAN_RE = re.compile(r"([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([NBRQnbrq]))?")
_UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


@dataclass(frozen=True, slots=True)
class Move:
    from_sq: int
    to_sq: int
    promotion: str = ""  # "q", "r", "b", "n" or ""

    def uci(self) -> str:
        return square_name(self.from_sq) + square_name(self.to_sq) + self.promotion

    @staticmethod
    def from_uci(text: str) -> Move:
        if len(text) not in (4, 5) or (len(text) == 5 and text[4] not in PROMOTIONS):
            raise ValueError(f"invalid UCI move '{text}'")
        return Move(parse_square(text[:2]), parse_square(text[2:4]), text[4:])

    def __str__(self) -> str:
        return self.uci()


class Position:
    """An immutable chess position. Build with from_fen() or start()."""

    __slots__ = ("board", "turn", "castling", "ep", "halfmove", "fullmove", "_legal")

    def __init__(self, board: str, turn: int, castling: str, ep: int | None, halfmove: int = 0, fullmove: int = 1):
        self.board = board
        self.turn = turn
        self.castling = castling  # a subset of "KQkq", in that order
        self.ep = ep  # the en passant square after any double push, as in FEN
        self.halfmove = halfmove
        self.fullmove = fullmove
        self._legal = None  # legal moves, computed on first use

    # --- FEN ---------------------------------------------------------------

    @staticmethod
    def start() -> Position:
        return Position.from_fen(START_FEN)

    @staticmethod
    def from_fen(fen: str, validate: bool = True) -> Position:
        """Parses a FEN; raises FenError. validate=False accepts positions
        that break the rules (for the board editor; see problem())."""
        fields = fen.split()
        if not 4 <= len(fields) <= 6:
            raise FenError("expected 4 to 6 fields")
        rows = fields[0].split("/")
        if len(rows) != 8:
            raise FenError("piece placement must describe 8 ranks")
        cells = [EMPTY] * 64
        for i, row in enumerate(rows):
            rank, file = 7 - i, 0
            for c in row:
                if c in "12345678":
                    file += int(c)
                elif c in "PNBRQKpnbrqk":
                    if file >= 8:
                        raise FenError(f"rank {rank + 1} has more than 8 squares")
                    cells[square(file, rank)] = c
                    file += 1
                else:
                    raise FenError(f"unknown piece letter '{c}'")
                if file > 8:
                    raise FenError(f"rank {rank + 1} has more than 8 squares")
            if file != 8:
                raise FenError(f"rank {rank + 1} does not have 8 squares")
        if fields[1] not in ("w", "b"):
            raise FenError("side to move must be 'w' or 'b'")
        turn = WHITE if fields[1] == "w" else BLACK
        castling = ""
        if fields[2] != "-":
            for c in fields[2]:
                if c not in "KQkq":
                    raise FenError(f"unknown castling letter '{c}'")
                if c in castling:
                    raise FenError(f"castling letter '{c}' repeated")
                castling += c
            castling = "".join(c for c in "KQkq" if c in castling)
        ep = None
        if fields[3] != "-":
            try:
                ep = parse_square(fields[3])
            except ValueError:
                raise FenError("invalid en passant square") from None
            if rank_of(ep) != (5 if turn == WHITE else 2):
                raise FenError("en passant square must be on rank 6 (white to move) or rank 3 (black to move)")
        counters = []
        for text, name in zip(fields[4:], ("halfmove clock", "fullmove number")):
            if not text.isdigit():
                raise FenError(f"invalid {name}")
            counters.append(int(text))
        halfmove = counters[0] if counters else 0
        fullmove = counters[1] if len(counters) > 1 else 1
        pos = Position("".join(cells), turn, castling, ep, halfmove, fullmove)
        problem = pos.problem() if validate else ""
        if problem:
            raise FenError(problem)
        return pos

    def problem(self) -> str:
        """Why this position is invalid, or "" if it's valid (the engine's rules)."""
        b = self.board
        if b.count("K") != 1 or b.count("k") != 1:
            return "each side needs exactly one king"
        if any(p in "Pp" for p in b[:8] + b[56:]):
            return "pawn on the first or last rank"
        for right, king_sq, rook_sq, king, rook in (("K", 4, 7, "K", "R"), ("Q", 4, 0, "K", "R"),
                                                    ("k", 60, 63, "k", "r"), ("q", 60, 56, "k", "r")):
            if right in self.castling and (b[king_sq] != king or b[rook_sq] != rook):
                return (f"castling right {right} needs the {'white' if king == 'K' else 'black'} king on "
                        f"{square_name(king_sq)} and a rook on {square_name(rook_sq)}")
        if self.ep is not None:
            # The opponent just pushed a pawn two squares: from `origin`, over
            # the en passant square, onto `pushed`.
            step = -8 if self.turn == WHITE else 8
            pushed, origin = self.ep + step, self.ep - step
            if b[self.ep] != EMPTY or b[origin] != EMPTY or b[pushed] != ("p" if self.turn == WHITE else "P"):
                return "en passant square does not follow a double pawn push"
        if self.is_attacked(self.king_square(other(self.turn)), self.turn):
            return "the side not to move is in check"
        return ""

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                p = self.board[square(file, rank)]
                if p == EMPTY:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                row += p
            if empty:
                row += str(empty)
            rows.append(row)
        ep = square_name(self.ep) if self.ep is not None else "-"
        return f"{'/'.join(rows)} {'wb'[self.turn]} {self.castling or '-'} {ep} {self.halfmove} {self.fullmove}"

    # --- Attacks -----------------------------------------------------------

    def king_square(self, color: int) -> int:
        return self.board.index("K" if color == WHITE else "k")

    def is_attacked(self, sq: int, by: int) -> bool:
        """Is `sq` attacked by a piece of color `by`?"""
        b = self.board
        f = file_of(sq)
        if by == WHITE:
            pawn, knight, bishop, rook, queen, king = "PNBRQK"
            if sq >= 8 and ((f > 0 and b[sq - 9] == pawn) or (f < 7 and b[sq - 7] == pawn)):
                return True
        else:
            pawn, knight, bishop, rook, queen, king = "pnbrqk"
            if sq < 56 and ((f > 0 and b[sq + 7] == pawn) or (f < 7 and b[sq + 9] == pawn)):
                return True
        for t in KNIGHT_TARGETS[sq]:
            if b[t] == knight:
                return True
        for t in KING_TARGETS[sq]:
            if b[t] == king:
                return True
        for ray in ROOK_RAYS[sq]:
            for t in ray:
                p = b[t]
                if p != EMPTY:
                    if p == rook or p == queen:
                        return True
                    break
        for ray in BISHOP_RAYS[sq]:
            for t in ray:
                p = b[t]
                if p != EMPTY:
                    if p == bishop or p == queen:
                        return True
                    break
        return False

    def in_check(self) -> bool:
        return self.is_attacked(self.king_square(self.turn), other(self.turn))

    # --- Moves -------------------------------------------------------------

    def _own(self, piece: str) -> bool:
        return piece != EMPTY and (piece.isupper() == (self.turn == WHITE))

    def _enemy(self, piece: str) -> bool:
        return piece != EMPTY and (piece.isupper() != (self.turn == WHITE))

    def pseudo_legal_moves(self) -> list[Move]:
        """Moves that follow the pieces' movement rules, including castling
        (not out of, through or into check); they may leave the king in check."""
        b = self.board
        moves = []
        for sq in range(64):
            p = b[sq]
            if not self._own(p):
                continue
            kind = p.upper()
            if kind == "P":
                self._pawn_moves(sq, moves)
            elif kind == "N" or kind == "K":
                for t in (KNIGHT_TARGETS if kind == "N" else KING_TARGETS)[sq]:
                    if not self._own(b[t]):
                        moves.append(Move(sq, t))
                if kind == "K":
                    self._castling_moves(sq, moves)
            else:
                rays = ()
                if kind != "B":
                    rays += ROOK_RAYS[sq]
                if kind != "R":
                    rays += BISHOP_RAYS[sq]
                for ray in rays:
                    for t in ray:
                        q = b[t]
                        if q == EMPTY:
                            moves.append(Move(sq, t))
                            continue
                        if self._enemy(q):
                            moves.append(Move(sq, t))
                        break
        return moves

    def _pawn_moves(self, sq: int, moves: list[Move]) -> None:
        b = self.board
        white = self.turn == WHITE
        step = 8 if white else -8
        last_rank = 7 if white else 0

        def add(to):
            if rank_of(to) == last_rank:
                moves.extend(Move(sq, to, p) for p in PROMOTIONS)
            else:
                moves.append(Move(sq, to))

        to = sq + step
        if b[to] == EMPTY:
            add(to)
            if rank_of(sq) == (1 if white else 6) and b[to + step] == EMPTY:
                moves.append(Move(sq, to + step))
        for df in (-1, 1):
            if 0 <= file_of(sq) + df < 8:
                to = sq + step + df
                if self._enemy(b[to]) or to == self.ep:
                    add(to)

    def _castling_moves(self, sq: int, moves: list[Move]) -> None:
        b = self.board
        white = self.turn == WHITE
        if sq != (4 if white else 60):
            return
        them = other(self.turn)
        kingside, queenside = ("K", "Q") if white else ("k", "q")
        if kingside in self.castling and b[sq + 1] == EMPTY and b[sq + 2] == EMPTY:
            if not any(self.is_attacked(s, them) for s in (sq, sq + 1, sq + 2)):
                moves.append(Move(sq, sq + 2))
        if queenside in self.castling and b[sq - 1] == EMPTY and b[sq - 2] == EMPTY and b[sq - 3] == EMPTY:
            if not any(self.is_attacked(s, them) for s in (sq, sq - 1, sq - 2)):
                moves.append(Move(sq, sq - 2))

    def legal_moves(self) -> list[Move]:
        if self._legal is None:
            us = self.turn
            legal = []
            for m in self.pseudo_legal_moves():
                after = self._apply(m)
                if not after.is_attacked(after.king_square(us), after.turn):
                    legal.append(m)
            self._legal = legal
        return self._legal

    def is_legal(self, move: Move) -> bool:
        return move in self.legal_moves()

    def is_capture(self, move: Move) -> bool:
        return self.board[move.to_sq] != EMPTY or self.is_en_passant(move)

    def is_en_passant(self, move: Move) -> bool:
        return (move.to_sq == self.ep and self.board[move.from_sq] in "Pp"
                and file_of(move.from_sq) != file_of(move.to_sq))

    def is_castling(self, move: Move) -> bool:
        return self.board[move.from_sq] in "Kk" and abs(move.to_sq - move.from_sq) == 2

    def push(self, move: Move) -> Position:
        """The position after a legal move; raises IllegalMoveError otherwise."""
        if not self.is_legal(move):
            raise IllegalMoveError(f"illegal move {move.uci()} in {self.fen()}")
        return self._apply(move)

    def _apply(self, m: Move) -> Position:
        b = list(self.board)
        piece = b[m.from_sq]
        captured = b[m.to_sq]
        b[m.to_sq] = piece
        b[m.from_sq] = EMPTY
        ep = None
        if piece in "Pp":
            if self.is_en_passant(m):
                b[m.to_sq + (-8 if piece == "P" else 8)] = EMPTY
                captured = "p" if piece == "P" else "P"
            elif abs(m.to_sq - m.from_sq) == 16:
                ep = (m.from_sq + m.to_sq) // 2
            if m.promotion:
                b[m.to_sq] = m.promotion.upper() if piece == "P" else m.promotion
        elif piece in "Kk" and abs(m.to_sq - m.from_sq) == 2:
            rook_from, rook_to = (m.from_sq + 3, m.from_sq + 1) if m.to_sq > m.from_sq else (m.from_sq - 4, m.from_sq - 1)
            b[rook_to] = b[rook_from]
            b[rook_from] = EMPTY
        castling = self.castling
        for sq in (m.from_sq, m.to_sq):
            lost = _CASTLING_SQUARES.get(sq)
            if lost:
                castling = "".join(c for c in castling if c not in lost)
        halfmove = 0 if piece in "Pp" or captured != EMPTY else self.halfmove + 1
        fullmove = self.fullmove + (1 if self.turn == BLACK else 0)
        return Position("".join(b), other(self.turn), castling, ep, halfmove, fullmove)

    # --- How games end -------------------------------------------------------

    def null_move(self) -> Position:
        """The position after a pass: the other side to move, no en passant
        (what that side would do with a free move: its threat). Raises
        IllegalMoveError when the side to move is in check."""
        if self.in_check():
            raise IllegalMoveError("no passing while in check")
        return Position(self.board, other(self.turn), self.castling, None, self.halfmove + 1,
                        self.fullmove + (1 if self.turn == BLACK else 0))

    def is_checkmate(self) -> bool:
        return not self.legal_moves() and self.in_check()

    def is_stalemate(self) -> bool:
        return not self.legal_moves() and not self.in_check()

    def is_insufficient_material(self) -> bool:
        """No sequence of legal moves can end in checkmate: no pawns, rooks or
        queens, and at most one minor piece, or only bishops, all on squares
        of one color."""
        minors = []
        for sq, p in enumerate(self.board):
            if p in "PpRrQq":
                return False
            if p in "NnBb":
                minors.append((p, sq))
        if len(minors) <= 1:
            return True
        if all(p in "Bb" for p, _ in minors):
            return len({(file_of(sq) + rank_of(sq)) & 1 for _, sq in minors}) == 1
        return False

    def has_legal_en_passant(self) -> bool:
        return self.ep is not None and any(self.is_en_passant(m) for m in self.legal_moves())

    def key(self) -> tuple:
        """What makes two positions the same for repetitions: pieces, side to
        move, castling rights and an en passant capture that is possible."""
        return (self.board, self.turn, self.castling, self.ep if self.has_legal_en_passant() else None)

    # --- Notation ------------------------------------------------------------

    def san(self, move: Move) -> str:
        """Standard algebraic notation of a legal move, with + or #."""
        if not self.is_legal(move):
            raise IllegalMoveError(f"illegal move {move.uci()} in {self.fen()}")
        piece = self.board[move.from_sq]
        kind = piece.upper()
        if self.is_castling(move):
            text = "O-O" if move.to_sq > move.from_sq else "O-O-O"
        elif kind == "P":
            text = (FILES[file_of(move.from_sq)] + "x" if self.is_capture(move) else "") + square_name(move.to_sq)
            if move.promotion:
                text += "=" + move.promotion.upper()
        else:
            rivals = [m.from_sq for m in self.legal_moves()
                      if m.to_sq == move.to_sq and m.from_sq != move.from_sq and self.board[m.from_sq] == piece]
            hint = ""
            if rivals:
                if all(file_of(s) != file_of(move.from_sq) for s in rivals):
                    hint = FILES[file_of(move.from_sq)]
                elif all(rank_of(s) != rank_of(move.from_sq) for s in rivals):
                    hint = str(rank_of(move.from_sq) + 1)
                else:
                    hint = square_name(move.from_sq)
            text = kind + hint + ("x" if self.is_capture(move) else "") + square_name(move.to_sq)
        after = self._apply(move)
        if after.in_check():
            text += "#" if not after.legal_moves() else "+"
        return text

    def parse_san(self, text: str) -> Move:
        """The legal move written as SAN; also accepts castling with zeros,
        missing or extra check marks, annotations (!, ?), long algebraic
        (e2-e4, Ng1f3) and UCI (e7e8q). Raises IllegalMoveError."""
        s = text.strip()
        while s and s[-1] in "+#!?":
            s = s[:-1]
        legal = self.legal_moves()
        if s in ("O-O", "0-0", "O-O-O", "0-0-0"):
            kingside = len(s) == 3
            for m in legal:
                if self.is_castling(m) and (m.to_sq > m.from_sq) == kingside:
                    return m
            raise IllegalMoveError(f"{text}: castling isn't legal in {self.fen()}")
        if _UCI_RE.fullmatch(s):
            move = Move.from_uci(s)
            if move in legal:
                return move
        match = _SAN_RE.fullmatch(s.replace("-", ""))
        if not match:
            raise IllegalMoveError(f"can't read move '{text}'")
        kind, from_file, from_rank, _, to, promo = match.groups()
        to_sq = parse_square(to)
        promo = (promo or "").lower()
        kind = kind or "P"
        found = []
        for m in legal:
            if m.to_sq != to_sq or self.board[m.from_sq].upper() != kind or m.promotion != promo:
                continue
            if from_file and file_of(m.from_sq) != FILES.index(from_file):
                continue
            if from_rank and rank_of(m.from_sq) != int(from_rank) - 1:
                continue
            found.append(m)
        if len(found) == 1:
            return found[0]
        if not found:
            raise IllegalMoveError(f"{text}: no such legal move in {self.fen()}")
        raise IllegalMoveError(f"{text}: ambiguous in {self.fen()}")

    def parse_uci(self, text: str) -> Move:
        """The legal move written in UCI notation; raises IllegalMoveError."""
        try:
            move = Move.from_uci(text)
        except ValueError as e:
            raise IllegalMoveError(str(e)) from None
        if not self.is_legal(move):
            raise IllegalMoveError(f"illegal move {text} in {self.fen()}")
        return move

    # --- Misc ------------------------------------------------------------------

    def __eq__(self, rhs: object) -> bool:
        return isinstance(rhs, Position) and self.fen() == rhs.fen()

    def __hash__(self) -> int:
        return hash(self.fen())

    def __repr__(self) -> str:
        return f"Position('{self.fen()}')"

    def ascii(self) -> str:
        return "\n".join(" ".join(self.board[square(f, r)] for f in range(8)) for r in range(7, -1, -1))


class Ending(enum.Enum):
    """Ways the rules end a game by themselves."""

    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient material"
    FIFTY_MOVES = "fifty-move rule"
    THREEFOLD_REPETITION = "threefold repetition"


def rules_ending(pos: Position, repetitions: int) -> Ending | None:
    """How the rules end the game in `pos`, if they do. `repetitions` counts
    how often pos.key() has occurred, including now. Checkmate and stalemate
    come first: a mate on the move that reaches 100 halfmoves still wins."""
    if not pos.legal_moves():
        return Ending.CHECKMATE if pos.in_check() else Ending.STALEMATE
    if pos.is_insufficient_material():
        return Ending.INSUFFICIENT_MATERIAL
    if pos.halfmove >= 100:
        return Ending.FIFTY_MOVES
    if repetitions >= 3:
        return Ending.THREEFOLD_REPETITION
    return None


def perft(pos: Position, depth: int) -> int:
    """Number of leaf nodes of the legal move tree, for testing."""
    if depth == 0:
        return 1
    moves = pos.legal_moves()
    if depth == 1:
        return len(moves)
    return sum(perft(pos._apply(m), depth - 1) for m in moves)
