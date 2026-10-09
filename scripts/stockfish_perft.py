"""Perft counts from Stockfish, used as an independent oracle.

  python scripts/stockfish_perft.py verify tests/perft/standard.epd
      Checks every expected count in a perft suite against Stockfish.
  python scripts/stockfish_perft.py counts "<FEN>" <max_depth>
      Prints an EPD line with Stockfish's counts for depths 1..max_depth.
  python scripts/stockfish_perft.py divide "<FEN>" <depth>
      Prints Stockfish's per-move counts ("e2e4: 20") and the total.

Stockfish defaults to tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe;
override with the STOCKFISH environment variable.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"


class Stockfish:
    def __init__(self, path=None):
        exe = path or os.environ.get("STOCKFISH") or str(DEFAULT_STOCKFISH)
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("uci")
        self._read_until(lambda line: line == "uciok")

    def _send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _read_until(self, done):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish exited unexpectedly")
            line = line.strip()
            lines.append(line)
            if done(line):
                return lines

    def divide(self, fen, depth, moves=()):
        """Returns ({uci_move: nodes}, total) for the position after `moves`."""
        self._send(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))
        self._send(f"go perft {depth}")
        lines = self._read_until(lambda line: line.startswith("Nodes searched:"))
        counts = {}
        for line in lines:
            if ": " in line and not line.startswith("Nodes searched"):
                move, count = line.split(": ")
                if count.isdigit():
                    counts[move] = int(count)
        total = int(lines[-1].split(":")[1])
        return counts, total

    def perft(self, fen, depth):
        return self.divide(fen, depth)[1]

    def close(self):
        self._send("quit")
        self.proc.wait(timeout=10)


def parse_epd_line(line):
    parts = [p.strip() for p in line.split(";")]
    expected = []
    for entry in parts[1:]:
        depth, nodes = entry.split()
        expected.append((int(depth[1:]), int(nodes)))
    return parts[0], expected


def verify(path):
    sf = Stockfish()
    failures = 0
    for number, raw in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fen, expected = parse_epd_line(line)
        for depth, nodes in expected:
            got = sf.perft(fen, depth)
            status = "ok  " if got == nodes else "FAIL"
            failures += got != nodes
            print(f"{status} line {number} D{depth}: expected {nodes}, stockfish {got}", flush=True)
    sf.close()
    print(f"{failures} mismatches")
    return 1 if failures else 0


def counts(fen, max_depth):
    sf = Stockfish()
    entries = " ".join(f";D{d} {sf.perft(fen, d)}" for d in range(1, max_depth + 1))
    sf.close()
    print(f"{fen} {entries}")
    return 0


def divide(fen, depth):
    sf = Stockfish()
    moves, total = sf.divide(fen, depth)
    sf.close()
    for move in sorted(moves):
        print(f"{move}: {moves[move]}")
    print(f"\nNodes searched: {total}")
    return 0


def main(argv):
    if len(argv) >= 2 and argv[0] == "verify":
        return verify(argv[1])
    if len(argv) >= 3 and argv[0] == "counts":
        return counts(argv[1], int(argv[2]))
    if len(argv) >= 3 and argv[0] == "divide":
        return divide(argv[1], int(argv[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
