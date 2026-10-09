"""A scripted UCI engine for testing the GUI's engine client and game
controller. It plays the first legal move in UCI order.

Flags: --name NAME, --no-uciok, --no-readyok, --slow-start SECONDS,
--crash-on-go N (exit abruptly on the Nth go), --illegal (move from an empty
square), --wait-for-stop (every search runs until "stop"), --delay MS (think
time before moving), --resign-none (answer "bestmove (none)"), --multipv
(offers MultiPV: line k is the k-th legal move and the first reply to it,
scored 12 - k cp), --blunder-at N (from ply N of a game from the start
position, the side that made move N stands lost: 500 cp for the other).
"""

import argparse
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chessgpu_gui.rules import BLACK, START_FEN, WHITE, Position, square_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Fake")
    ap.add_argument("--no-uciok", action="store_true")
    ap.add_argument("--no-readyok", action="store_true")
    ap.add_argument("--slow-start", type=float, default=0)
    ap.add_argument("--crash-on-go", type=int, default=0)
    ap.add_argument("--illegal", action="store_true")
    ap.add_argument("--wait-for-stop", action="store_true")
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--resign-none", action="store_true")
    ap.add_argument("--multipv", action="store_true")
    ap.add_argument("--blunder-at", type=int, default=0)
    a = ap.parse_args()
    multipv = [1]

    lock = threading.Lock()

    def say(line):
        with lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    pos = Position.start()
    stop = threading.Event()
    thinker = None
    gos = 0

    def score_of(p, cp):
        """`cp`, or the scripted blunder's score (side to move's view)."""
        ply = 2 * (p.fullmove - 1) + (1 if p.turn == BLACK else 0)
        if not a.blunder_at or ply < a.blunder_at:
            return cp
        blunderer = WHITE if (a.blunder_at - 1) % 2 == 0 else BLACK
        white_view = -500 if blunderer == WHITE else 500
        return white_view if p.turn == WHITE else -white_view

    def think(p, infinite):
        legal = sorted(m.uci() for m in p.legal_moves())
        if a.illegal:
            empty = next(s for s in range(64) if p.board[s] == ".")
            move = square_name(empty) + ("a1" if empty != 0 else "b1")
        elif a.resign_none or not legal:
            move = None
        else:
            move = legal[0]
        say(f"info depth 1 seldepth 2 score cp {score_of(p, 12)} nodes 100 nps 1000 time 1 pv {move or ''}".rstrip())
        if a.multipv and multipv[0] > 1:
            for k, first in enumerate(legal[:multipv[0]], start=1):
                after = p.push(p.parse_uci(first))
                replies = sorted(m.uci() for m in after.legal_moves())
                pv = " ".join([first] + replies[:1])
                say(f"info depth 1 multipv {k} score cp {12 - k} nodes 100 nps 1000 time 1 pv {pv}")
        if infinite or a.wait_for_stop:
            stop.wait()
        elif a.delay:
            stop.wait(a.delay / 1000)
        say(f"info depth 2 score cp {score_of(p, 15)} nodes 300 nps 3000 time 2 pv {move or ''}".rstrip())
        say(f"bestmove {move or '(none)'}")

    for line in sys.stdin:
        words = line.split()
        if not words:
            continue
        cmd = words[0]
        if cmd == "uci":
            if a.slow_start:
                time.sleep(a.slow_start)
            say(f"id name {a.name}")
            say("id author the tests")
            say("option name Hash type spin default 16 min 1 max 1024")
            say("option name Style type combo default Normal var Normal var Wild var Calm Mode")
            say("option name Ponder type check default false")
            say("option name Clear Hash type button")
            say("option name Book File type string default <empty>")
            if a.multipv:
                say("option name MultiPV type spin default 1 min 1 max 500")
            if not a.no_uciok:
                say("uciok")
        elif cmd == "isready":
            if not a.no_readyok:
                say("readyok")
        elif cmd == "setoption":
            say("info string set " + " ".join(words[2:]))
            if len(words) >= 5 and words[2].lower() == "multipv" and words[3] == "value":
                multipv[0] = int(words[4])
        elif cmd == "ucinewgame":
            say("info string newgame")
        elif cmd == "position":
            if words[1] == "startpos":
                fen, rest = START_FEN, words[2:]
            else:
                end = words.index("moves") if "moves" in words else len(words)
                fen, rest = " ".join(words[2:end]), words[end:]
            pos = Position.from_fen(fen)
            for m in rest[1:]:
                pos = pos.push(pos.parse_uci(m))
        elif cmd == "go":
            gos += 1
            if a.crash_on_go and gos >= a.crash_on_go:
                sys.stdout.flush()
                import os
                os._exit(3)
            stop.clear()
            thinker = threading.Thread(target=think, args=(pos, "infinite" in words))
            thinker.start()
        elif cmd == "stop":
            stop.set()
            if thinker:
                thinker.join()
        elif cmd == "quit":
            stop.set()
            if thinker:
                thinker.join()
            break


if __name__ == "__main__":
    main()
