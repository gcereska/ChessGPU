"""Shared test helpers: the perft suites and a minimal UCI client for
cross-checking the GUI's rules against the engine (independent of the GUI's
own Qt-based client in chessgpu_gui/uci.py)."""

import os
import pathlib
import subprocess

GUI_DIR = pathlib.Path(__file__).resolve().parent.parent
ROOT = GUI_DIR.parent
PERFT_DIR = ROOT / "tests" / "perft"


def engine_path():
    env = os.environ.get("CHESSGPU_ENGINE")
    path = pathlib.Path(env) if env else ROOT / "build" / "Release" / "chessgpu.exe"
    return path if path.exists() else None


def read_perft_suite(name):
    """[(fen, {depth: nodes})] from tests/perft/<name>."""
    out = []
    for line in (PERFT_DIR / name).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(";")]
        counts = {int(f.split()[0][1:]): int(f.split()[1]) for f in fields[1:]}
        out.append((fields[0], counts))
    return out


def all_perft_positions():
    return read_perft_suite("standard.epd") + read_perft_suite("edge_cases.epd")


class SimpleEngine:
    """Talks to the engine over pipes. Only uses commands that don't start a
    search backend (no isready, no go), so it never touches the GPU."""

    def __init__(self, path):
        self.p = subprocess.Popen([str(path), "uci"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1)
        self.send("uci")
        self.read_until("uciok")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read_until(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def set_position(self, fen, moves=()):
        self.send(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))

    def legal_moves(self):
        self.send("legalmoves")
        return set(self.read_until("info string legalmoves").split()[3:])

    def fen(self):
        self.send("d")
        return self.read_until("info string fen ")[len("info string fen "):]

    def close(self):
        self.send("quit")
        self.p.wait(timeout=10)
