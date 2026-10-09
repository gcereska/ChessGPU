"""PGN reading and writing.

A game is a tree of PgnNodes under its root (the start position): a node's
first child continues its line, the others are variations, alternatives to
the first child. `moves` and `comments` give the main line.

Writing follows the PGN export format: the Seven Tag Roster first, SetUp and
FEN for games that don't start from the initial position, SAN movetext with
move numbers, NAGs ($1), comments in braces, each variation in parentheses
right after the move it replaces, lines of at most 80 characters.

Reading is tolerant: any tag order, comments ({...} and ;...), NAGs and
annotation marks (!, ?, !!, ??, !?, ?! become NAGs 1-6), escape lines (%...),
missing results, and several games per file. Main-line moves must be legal;
a variation with an illegal or null move is dropped from that move on.

Comments may carry commands, as lichess and ChessBase write them: arrows
[%cal Ge2e4,Rd8d1], circles [%csl Gd5], the clock [%clk 0:03:12] and an
evaluation [%eval 0.35]. split_commands() and join_commands() read and write
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rules import START_FEN, WHITE, FenError, IllegalMoveError, Move, Position

RESULTS = ("1-0", "0-1", "1/2-1/2", "*")
SEVEN_TAG_ROSTER = ("Event", "Site", "Date", "Round", "White", "Black", "Result")
_ROSTER_DEFAULTS = {"Event": "?", "Site": "?", "Date": "????.??.??", "Round": "?", "White": "?", "Black": "?"}
LINE_LENGTH = 80
GLYPH_NAGS = {"!": 1, "?": 2, "!!": 3, "??": 4, "!?": 5, "?!": 6}


class PgnError(ValueError):
    """PGN text that can't be read: an illegal move, a bad FEN, broken syntax."""


@dataclass(eq=False)
class PgnNode:
    """A move of a game tree (the root: no move), what was said about it, and
    what follows it."""

    move: Move | None = None
    comment: str = ""  # after the move (the root's: before the first move)
    nags: list[int] = field(default_factory=list)
    starting_comment: str = ""  # before the move, at the start of a variation
    children: list[PgnNode] = field(default_factory=list, repr=False)

    def add(self, move: Move, comment: str = "") -> PgnNode:
        child = PgnNode(move, comment)
        self.children.append(child)
        return child


class PgnGame:
    """Tags, the start position, the result and the tree of moves."""

    def __init__(self, tags: dict[str, str] | None = None, start: Position | None = None,
                 moves: list[Move] | tuple = (), comments: list[str] | tuple = (), result: str = "*"):
        self.tags: dict[str, str] = dict(tags or {})
        self.start = start if start is not None else Position.start()
        self.result = result
        self.root = PgnNode()
        node = self.root
        for i, move in enumerate(moves):
            node = node.add(move, comments[i] if i < len(comments) else "")

    def mainline(self) -> list[PgnNode]:
        """The main line's nodes (the root excluded)."""
        out, node = [], self.root
        while node.children:
            node = node.children[0]
            out.append(node)
        return out

    @property
    def moves(self) -> list[Move]:
        return [n.move for n in self.mainline()]

    @property
    def comments(self) -> list[str]:
        return [n.comment for n in self.mainline()]

    def comment(self, i: int) -> str:
        line = self.mainline()
        return line[i].comment if i < len(line) else ""

    def positions(self) -> list[Position]:
        """The start position and the position after every main-line move."""
        out = [self.start]
        for m in self.moves:
            out.append(out[-1].push(m))
        return out


# --- Comment commands ---------------------------------------------------------------

_COMMAND_RE = re.compile(r"\[%(?P<name>[A-Za-z]\w*)\s*(?P<value>[^\]]*)\]")


def split_commands(comment: str) -> tuple[str, dict[str, str]]:
    """The comment's text without its [%name value] commands (whitespace
    tidied), and the commands by name (a repeated name: the values joined
    with commas, as for [%cal])."""
    commands: dict[str, str] = {}

    def take(m):
        name, value = m.group("name"), m.group("value").strip()
        if value:
            commands[name] = f"{commands[name]},{value}" if commands.get(name) else value
        else:
            commands.setdefault(name, "")
        return " "

    text = _COMMAND_RE.sub(take, comment)
    return " ".join(text.split()), commands


def join_commands(text: str, commands: dict[str, str]) -> str:
    """Comment text followed by its commands (empty values left out)."""
    parts = [text.strip()] if text.strip() else []
    parts += [f"[%{name} {value}]" for name, value in commands.items() if value]
    return " ".join(parts)


# --- Writing ----------------------------------------------------------------------------


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _unescape(value: str) -> str:
    return re.sub(r"\\(.)", r"\1", value)


def _comment_tokens(text: str) -> list[str]:
    """A comment as tokens the line wrapper may break between ("}" in the
    text becomes ")": a comment can't contain its own end)."""
    words = text.replace("}", ")").split()
    if not words:
        return []
    words[0] = "{" + words[0]
    words[-1] += "}"
    return words


def _move_tokens(node: PgnNode, pos: Position, number: bool) -> list[str]:
    """A move with its number (always for White; for Black when `number`),
    NAGs and comment."""
    tokens = []
    if pos.turn == WHITE:
        tokens.append(f"{pos.fullmove}.")
    elif number:
        tokens.append(f"{pos.fullmove}...")
    tokens.append(pos.san(node.move))
    tokens += [f"${n}" for n in node.nags]
    return tokens + _comment_tokens(node.comment)


def _line_tokens(node: PgnNode, pos: Position, number: bool, tokens: list[str]) -> None:
    """The moves after `node` (whose position is `pos`): the line it
    continues in, each move followed by its variations."""
    while node.children:
        main = node.children[0]
        tokens += _move_tokens(main, pos, number)
        number = bool(main.comment.strip())  # Black's move after a comment needs its number again
        for variation in node.children[1:]:
            first = len(tokens)
            tokens += _comment_tokens(variation.starting_comment)
            tokens += _move_tokens(variation, pos, True)
            _line_tokens(variation, pos.push(variation.move), bool(variation.comment.strip()), tokens)
            tokens[first] = "(" + tokens[first]
            tokens[-1] += ")"
            number = True
        node, pos = main, pos.push(main.move)


def write_pgn(game: PgnGame) -> str:
    """One game as PGN text, ending with a newline."""
    if game.result not in RESULTS:
        raise PgnError(f"invalid result '{game.result}'")
    tags = {k: v for k, v in game.tags.items() if k not in ("SetUp", "FEN")}
    tags["Result"] = game.result
    if game.start.fen() != START_FEN:
        tags["SetUp"] = "1"
        tags["FEN"] = game.start.fen()
    ordered = [(k, tags.get(k, _ROSTER_DEFAULTS.get(k, "?"))) for k in SEVEN_TAG_ROSTER]
    ordered += [(k, v) for k, v in tags.items() if k not in SEVEN_TAG_ROSTER]
    header = "".join(f'[{k} "{_escape(str(v))}"]\n' for k, v in ordered)

    tokens = _comment_tokens(game.root.comment)
    _line_tokens(game.root, game.start, True, tokens)
    tokens.append(game.result)

    lines, line = [], ""
    for t in tokens:
        if line and len(line) + 1 + len(t) > LINE_LENGTH:
            lines.append(line)
            line = t
        else:
            line = f"{line} {t}" if line else t
    lines.append(line)
    return header + "\n" + "\n".join(lines) + "\n"


def write_pgns(games: list[PgnGame]) -> str:
    return "\n".join(write_pgn(g) for g in games)


# --- Reading ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
      \[\s*(?P<tag>[A-Za-z0-9_]+)\s+"(?P<value>(?:[^"\\]|\\.)*)"\s*\]
    | \{(?P<comment>[^}]*)\}
    | ;(?P<line_comment>[^\n]*)
    | (?P<open>\()
    | (?P<close>\))
    | \$(?P<nag>\d+)
    | (?P<result>1-0|0-1|1/2-1/2|\*)
    | (?P<number>\d+\s*\.+)
    | (?P<move>--|[A-Za-z0-9][A-Za-z0-9=+#!?\-]*)
    | (?P<annotation>[!?]+)
    | (?P<junk>\S)
""", re.VERBOSE)

_GLYPH_RE = re.compile(r"^(?P<san>.*?)(?P<glyph>[!?]*)$")


class _Reader:
    """The state while reading movetext: the game, the node the next move
    follows (`line`), the node comments and NAGs go to (`last`: the last
    move read; the root before the first move; None at the start of a
    variation), and the lines left open by "(" (a stack)."""

    def __init__(self):
        self.games: list[PgnGame] = []
        self.tags: dict[str, str] = {}
        self.game: PgnGame | None = None
        self.line: PgnNode | None = None
        self.last: PgnNode | None = None
        self.stack: list[tuple[PgnNode, PgnNode | None]] = []  # (line, last) at each "("
        self.parent: dict[int, PgnNode] = {}  # id(node) -> the node it follows
        self.position: dict[int, Position] = {}  # id(node) -> the position after it
        self.skip = 0  # > 0: the innermost variation is being dropped (counting the "(" inside it)
        self.pending = ""  # a comment before the next move (the game's or a variation's first)

    def where(self) -> str:
        return f"game {len(self.games) + 1}"

    def begin(self) -> None:
        try:
            start = Position.from_fen(self.tags["FEN"]) if "FEN" in self.tags else Position.start()
        except FenError as e:
            raise PgnError(f"{self.where()}: bad FEN tag: {e}") from None
        self.game = PgnGame(tags=dict(self.tags), start=start)
        self.game.root.comment, self.pending = self.pending, ""
        self.line = self.last = self.game.root
        self.parent = {}
        self.position = {id(self.game.root): start}

    def finish(self, result: str) -> None:
        if self.game is None:
            self.begin()
        self.game.result = result
        self.games.append(self.game)
        self.game, self.line, self.last, self.tags = None, None, None, {}
        self.stack, self.skip, self.pending = [], 0, ""

    def comment(self, text: str) -> None:
        text = " ".join(text.split())
        if not text or self.skip:
            return
        if self.last is None or self.game is None:
            self.pending = f"{self.pending} {text}".strip()  # before a move: a variation's or the game's first
        else:
            self.last.comment = f"{self.last.comment} {text}".strip()

    def nag(self, n: int) -> None:
        if not self.skip and self.last is not None and self.game is not None and self.last.move is not None:
            self.last.nags.append(n)

    def move(self, token: str) -> None:
        if self.skip:
            return
        if self.game is None:
            self.begin()
        m = _GLYPH_RE.match(token)
        san, glyph = m.group("san"), m.group("glyph")
        pos = self.position[id(self.line)]
        try:
            if san == "--":
                raise IllegalMoveError("null moves aren't supported")
            move = pos.parse_san(san)
        except IllegalMoveError as e:
            if not self.stack:
                raise PgnError(f"{self.where()}, move {pos.fullmove}: {e}") from None
            self.skip = 1  # drop the rest of this variation
            return
        node = PgnNode(move)
        if self.last is None:
            node.starting_comment, self.pending = self.pending, ""
        self.line.children.append(node)
        self.parent[id(node)] = self.line
        self.position[id(node)] = pos.push(move)
        self.line = self.last = node
        if glyph in GLYPH_NAGS:
            node.nags.append(GLYPH_NAGS[glyph])

    def open(self) -> None:
        if self.skip:
            self.skip += 1  # a variation inside one being dropped
            return
        if self.game is None:
            self.begin()
        self.stack.append((self.line, self.last))
        if self.last is None or self.last.move is None:
            self.skip = 1  # no move to be an alternative to: dropped
            return
        self.line, self.last, self.pending = self.parent[id(self.last)], None, ""

    def close(self) -> None:
        if self.skip > 1:
            self.skip -= 1
            return
        # The end of a variation, maybe one dropped part-way (what was read of it stays).
        self.skip = 0
        if not self.stack:
            raise PgnError(f"{self.where()}: ')' without '('")
        self.line, self.last = self.stack.pop()
        self.pending = ""


def read_pgn(text: str) -> list[PgnGame]:
    """Every game in `text`; raises PgnError on the first bad one."""
    text = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
    r = _Reader()
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "value":  # (only matched together with "tag")
            kind = "tag"
        if kind == "tag":
            if r.game is not None:  # tags of a new game: the last one had no result
                result = r.game.tags.get("Result", "*")
                r.finish(result if result in RESULTS else "*")
            elif not r.tags:
                r.pending = ""  # a comment before the tags belongs to no game
            r.tags[m.group("tag")] = _unescape(m.group("value"))
        elif kind == "comment":
            r.comment(m.group("comment"))
        elif kind == "line_comment":
            r.comment(m.group("line_comment"))
        elif kind == "nag":
            r.nag(int(m.group("nag")))
        elif kind == "annotation":
            if m.group("annotation") in GLYPH_NAGS:
                r.nag(GLYPH_NAGS[m.group("annotation")])
        elif kind == "number":
            pass
        elif kind == "open":
            r.open()
        elif kind == "close":
            r.close()
        elif kind == "result":
            if not r.stack and not r.skip:
                r.finish(m.group("result"))
        elif kind == "move":
            r.move(m.group("move"))
        else:
            raise PgnError(f"{r.where()}: unexpected '{m.group()}'")
    if r.game is not None or r.tags:
        result = (r.game.tags if r.game else r.tags).get("Result", "*")
        r.finish(result if result in RESULTS else "*")
    return r.games
