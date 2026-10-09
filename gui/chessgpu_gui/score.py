"""Engine scores and the move comments that carry them in PGN files
("+0.35/12 1.234s": score in pawns, depth, time), as written by fastchess
and cutechess and by this GUI."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Scores beyond this many centipawns count as decided for display (eval bar).
DECIDED_CP = 1000


@dataclass(frozen=True)
class Score:
    """A UCI score from one side's point of view (the engine's: the side to
    move): centipawns, or mate in `mate` moves (negative: getting mated).
    mate == 0 is the side to move mated now; its mirror image, needed when
    turning that into the other side's view, sets `given` (python-chess
    calls it MateGiven)."""

    cp: int | None = None
    mate: int | None = None
    given: bool = False

    def __post_init__(self):
        if (self.cp is None) == (self.mate is None):
            raise ValueError("a score is either centipawns or mate")
        if self.given and self.mate != 0:
            raise ValueError("only mate 0 can be given")

    def winning(self) -> bool:
        """A mate for this side (including mate given)."""
        return self.mate is not None and (self.mate > 0 or self.given)

    def negated(self) -> Score:
        """The same score from the other side's point of view."""
        if self.mate is not None:
            return Score(mate=0, given=not self.given) if self.mate == 0 else Score(mate=-self.mate)
        return Score(cp=-self.cp)

    def text(self) -> str:
        """+0.35, -1.20, 0.00, +M3, -M2 (-M0: mated now, +M0: mate given)."""
        if self.mate is not None:
            return ("+M" if self.winning() else "-M") + str(abs(self.mate))
        return "0.00" if self.cp == 0 else f"{self.cp / 100:+.2f}"

    def bar_cp(self) -> int:
        """Centipawns for an eval bar or graph: mates count as +-DECIDED_CP."""
        if self.mate is not None:
            return DECIDED_CP if self.winning() else -DECIDED_CP
        return max(-DECIDED_CP, min(DECIDED_CP, self.cp))


@dataclass(frozen=True)
class MoveEval:
    """What an engine reported for the move it played."""

    score: Score | None = None
    depth: int | None = None
    seconds: float | None = None

    def comment(self) -> str:
        """The PGN comment, e.g. "+0.35/12 1.234s"."""
        parts = []
        if self.score is not None:
            parts.append(self.score.text() + (f"/{self.depth}" if self.depth is not None else ""))
        if self.seconds is not None:
            parts.append(f"{self.seconds:.3f}s")
        return " ".join(parts)


_COMMENT_RE = re.compile(r"^\s*(?P<sign>[+-]?)(?P<mate>M?)(?P<value>\d+(?:\.\d+)?)/(?P<depth>\d+)"
                         r"(?:\s+(?P<seconds>\d+(?:\.\d+)?)s)?")
_SECONDS_RE = re.compile(r"^\s*(?P<seconds>\d+(?:\.\d+)?)s\s*$")


def parse_eval_comment(comment: str) -> MoveEval | None:
    """Reads "+0.35/12 1.234s" (anything after it is ignored), or just the
    time "1.234s" (a human's move); None if the comment starts with neither."""
    m = _COMMENT_RE.match(comment)
    if not m:
        t = _SECONDS_RE.match(comment)
        return MoveEval(None, None, float(t.group("seconds"))) if t else None
    negative = m.group("sign") == "-"
    if m.group("mate"):
        moves = int(float(m.group("value")))
        score = Score(mate=0, given=not negative) if moves == 0 else Score(mate=-moves if negative else moves)
    else:
        cp = round(float(m.group("value")) * 100)
        score = Score(cp=-cp if negative else cp)
    seconds = float(m.group("seconds")) if m.group("seconds") else None
    return MoveEval(score, int(m.group("depth")), seconds)


@dataclass(frozen=True)
class PositionEval:
    """An engine's evaluation of a position (game review): the score from
    White's point of view, the depth, and the best line (UCI moves)."""

    score: Score
    depth: int | None = None
    pv: tuple[str, ...] = ()

    def command(self) -> str:
        """The [%eval] value: "0.35", "-1.20", "#3" (White mates in 3),
        "#-2"; ",depth" after it when known."""
        s = self.score
        if s.mate is not None:
            text = f"#{'' if s.winning() else '-'}{abs(s.mate)}"
        else:
            text = f"{s.cp / 100:.2f}"
        return text + (f",{self.depth}" if self.depth is not None else "")


_EVAL_COMMAND_RE = re.compile(r"^\s*(?:#(?P<mate>[+-]?\d+)|(?P<pawns>[+-]?\d+(?:\.\d+)?))\s*(?:,\s*(?P<depth>\d+))?\s*$")


def parse_eval_command(text: str) -> PositionEval | None:
    """Reads an [%eval] value (see PositionEval.command); None if it isn't one."""
    m = _EVAL_COMMAND_RE.match(text)
    if not m:
        return None
    if m.group("mate") is not None:
        n = int(m.group("mate"))
        if n == 0:
            score = Score(mate=0, given=not m.group("mate").startswith("-"))
        else:
            score = Score(mate=n)
    else:
        score = Score(cp=round(float(m.group("pawns")) * 100))
    return PositionEval(score, int(m.group("depth")) if m.group("depth") else None)
