"""Turns match PGNs into a Texel tuning set: one line per position,
"<FEN> | <result> | <game>", the result from white's point of view (1, 0.5
or 0) and the game's number (the tuner keeps every fifth game for validation).

  python scripts/make_tuning_set.py [--out build/tuning/positions.txt] [--skip 8] [PGN ...]

By default it reads the PGNs in build/matches/ (our engine's own match
games; not the smoke games). The first --skip plies of each game (its opening) and
games without a result are left out; `chessgpu tune` filters the rest down
to quiet positions. PGNs are read with the GUI's reader (pure Python).
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gui"))

from chessgpu_gui.pgn import read_pgn  # noqa: E402

RESULT_VALUE = {"1-0": "1", "0-1": "0", "1/2-1/2": "0.5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="*", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "build" / "tuning" / "positions.txt")
    ap.add_argument("--skip", type=int, default=8)
    a = ap.parse_args()
    # build/matches/corpus/ holds copies of some top-level PGNs (not read
    # again); smoke games are fast self-play, left out.
    pgns = a.pgns or sorted(p for p in (ROOT / "build" / "matches").glob("*.pgn") if p.stem != "smoke")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    games = positions = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for path in pgns:
            for game in read_pgn(path.read_text(encoding="utf-8")):
                value = RESULT_VALUE.get(game.result)
                if value is None:
                    continue
                games += 1
                for i, pos in enumerate(game.positions()):
                    if i < a.skip:
                        continue
                    out.write(f"{pos.fen()} | {value} | {games}\n")
                    positions += 1
            print(f"{path}: {games} games, {positions} positions so far", file=sys.stderr)
    print(f"{games} games, {positions} positions -> {a.out}")


if __name__ == "__main__":
    main()
