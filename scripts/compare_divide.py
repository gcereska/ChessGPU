"""Compare our perft divide with Stockfish's and drill down to the first
position where the legal move lists differ.

  python scripts/compare_divide.py "<FEN>" <depth> [--model cpu|thread|warp] [--board std|quad]
                                                 [--engine PATH]

At each step it follows the first root move whose count differs, until it
reaches a position where the two programs disagree on the legal moves
themselves, and prints the moves leading there with the missing and extra
moves. Both programs always get the root FEN plus the move list, so state a
FEN can't express (like an en passant square Stockfish omits because the
capture is illegal) is preserved.
"""
import argparse
import subprocess
import sys

from stockfish_perft import ROOT, Stockfish

DEFAULT_ENGINE = ROOT / "build" / "Release" / "chessgpu.exe"


def our_divide(engine, fen, moves, depth, model, board):
    result = subprocess.run(
        [str(engine), "perft", str(depth), "--fen", fen, "--moves", " ".join(moves), "--divide", "--model", model,
         "--board", board],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"chessgpu failed: {result.stderr.strip()}")
    moves = {}
    for line in result.stdout.splitlines():
        if ": " in line and not line.startswith(("Nodes searched", "perft")):
            move, count = line.split(": ")
            moves[move] = int(count)
    return moves


def fen_after(sf, fen, moves):
    """Stockfish's FEN for the position after `moves` (for display only)."""
    sf._send(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))
    sf._send("d")
    lines = sf._read_until(lambda line: line.startswith("Checkers"))
    return next(line[len("Fen: "):] for line in lines if line.startswith("Fen: "))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fen")
    parser.add_argument("depth", type=int)
    parser.add_argument("--model", default="cpu")
    parser.add_argument("--board", default="quad")
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE))
    args = parser.parse_args()

    sf = Stockfish()
    fen, depth, path = args.fen, args.depth, []
    try:
        while depth >= 1:
            ours = our_divide(args.engine, fen, path, depth, args.model, args.board)
            theirs, _ = sf.divide(fen, depth, path)
            missing = sorted(set(theirs) - set(ours))
            extra = sorted(set(ours) - set(theirs))
            if missing or extra:
                print(f"Move lists differ after: {' '.join(path) or '(root position)'}")
                print(f"  root FEN: {fen}")
                print(f"  position (Stockfish FEN): {fen_after(sf, fen, path)}")
                print(f"  missing (Stockfish has them, we don't): {' '.join(missing) or '-'}")
                print(f"  extra (we have them, Stockfish doesn't): {' '.join(extra) or '-'}")
                return 1
            differing = [m for m in sorted(theirs) if theirs[m] != ours[m]]
            if not differing:
                if path:
                    print("Counts match here but differed one level up: unexpected, investigate by hand.")
                    return 1
                print(f"All {len(theirs)} root moves match at depth {depth}.")
                return 0
            move = differing[0]
            print(f"depth {depth}: {move} differs (ours {ours[move]}, Stockfish {theirs[move]})")
            path.append(move)
            depth -= 1
        print("Reached depth 0 without isolating the difference.")
        return 1
    finally:
        sf.close()


if __name__ == "__main__":
    sys.exit(main())
