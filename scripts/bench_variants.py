"""Runs `chessgpu bench-search` for several configurations and summarizes
the time to depth over the suite.

  python scripts/bench_variants.py <name> [--depth D] [--limit N] [--passes P] [--engine EXE]

<name> picks a list of configurations from CONFIGS below. Each run prints
its per-position lines to stderr and one summary line per configuration to
stdout; passes alternate the order of the configurations (A-B ... B-A, then
A-B again) and the best total is kept, as for perft (docs/ARCHITECTURE.md,
Results). With more than two passes, every pass's total is listed too, to show
the noise. --engine runs another build (e.g. a saved copy) instead of
build/Release/chessgpu.exe.
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "Release" / "chessgpu.exe"

DEVICE = ["--search", "ybwc-device", "--leaf-depth", "2", "--node-budget", "2048", "--cut-waves"]
ABDADA = ["--search", "abdada-device", "--features", "tt-killers-see"]

CONFIGS = {
    # The search algorithms, best known settings for each.
    "algorithms": [
        ("cpu", ["--search", "cpu"]),
        ("ybwc-host waves", ["--search", "ybwc-host", "--leaf-depth", "3", "--node-budget", "16384", "--cut-waves"]),
        ("ybwc-device waves", DEVICE),
        ("ybwc-device classic", ["--search", "ybwc-device", "--leaf-depth", "3", "--node-budget", "4096"]),
    ],
    # Execution models and thread-model implementations, under the device YBWC.
    "models": [
        ("thread staged local", DEVICE),
        ("thread staged shared", DEVICE + ["--stack", "shared"]),
        ("thread list local", DEVICE + ["--ordering", "list"]),
        ("warp block 64", DEVICE + ["--model", "warp", "--block-size", "64"]),
        ("warp block 32", DEVICE + ["--model", "warp", "--block-size", "32"]),
    ],
    "boards": [
        ("quad", DEVICE + ["--board", "quad"]),
        ("std", DEVICE + ["--board", "std"]),
    ],
    "evals": [
        ("pesto", DEVICE + ["--eval", "pesto"]),
        ("material", DEVICE + ["--eval", "material"]),
        ("terms", DEVICE + ["--eval", "terms"]),
    ],
    # YBWC settings of the device search.
    "ybwc": [
        ("leaf 2 budget 2048 waves", DEVICE),
        ("leaf 1 budget 1024 waves", ["--search", "ybwc-device", "--leaf-depth", "1", "--node-budget", "1024", "--cut-waves"]),
        ("leaf 2 budget 4096 waves", ["--search", "ybwc-device", "--leaf-depth", "2", "--node-budget", "4096", "--cut-waves"]),
        ("leaf 3 budget 2048 waves", ["--search", "ybwc-device", "--leaf-depth", "3", "--node-budget", "2048", "--cut-waves"]),
        ("leaf 2 budget 2048 classic", ["--search", "ybwc-device", "--leaf-depth", "2", "--node-budget", "2048"]),
        ("leaf 2 budget 2048 waves lanes 2", DEVICE + ["--lanes", "2"]),
        ("leaf 2 budget 2048 waves lanes 4", DEVICE + ["--lanes", "4"]),
    ],
    # Transposition table: feature sets and entry layouts.
    "tt": [
        ("none", DEVICE),
        ("tt packed64", DEVICE + ["--features", "tt", "--tt-layout", "packed64"]),
        ("tt wide128", DEVICE + ["--features", "tt", "--tt-layout", "wide128"]),
        ("tt-noqs packed64", DEVICE + ["--features", "tt-noqs", "--tt-layout", "packed64"]),
        ("tt-noqs wide128", DEVICE + ["--features", "tt-noqs", "--tt-layout", "wide128"]),
        ("tt-order packed64", DEVICE + ["--features", "tt-order", "--tt-layout", "packed64"]),
        ("tt-order wide128", DEVICE + ["--features", "tt-order", "--tt-layout", "wide128"]),
    ],
    # The feature sets whose table can end a search, in both layouts: the
    # close contenders of the "tt" group, for longer runs.
    "tt-cutoffs": [
        ("tt packed64", DEVICE + ["--features", "tt", "--tt-layout", "packed64"]),
        ("tt wide128", DEVICE + ["--features", "tt", "--tt-layout", "wide128"]),
        ("tt-noqs packed64", DEVICE + ["--features", "tt-noqs", "--tt-layout", "packed64"]),
        ("tt-noqs wide128", DEVICE + ["--features", "tt-noqs", "--tt-layout", "wide128"]),
    ],
    # Null-move pruning against the table alone, and its settings
    # (default: R = 3 + depth / 6, from depth 2).
    "nmp": [
        ("tt", DEVICE + ["--features", "tt"]),
        ("tt-nmp", DEVICE + ["--features", "tt-nmp"]),
        ("tt-nmp R 2", DEVICE + ["--features", "tt-nmp", "--null-reduction", "2", "--null-divisor", "0"]),
        ("tt-nmp R 3", DEVICE + ["--features", "tt-nmp", "--null-divisor", "0"]),
        ("tt-nmp R 4+d/6", DEVICE + ["--features", "tt-nmp", "--null-reduction", "4"]),
        ("tt-nmp from depth 3", DEVICE + ["--features", "tt-nmp", "--null-min-depth", "3"]),
    ],
    # Late move reductions against null moves alone, and their
    # settings (default: R = 0.75 + ln(depth) ln(move) / 2.25, from depth 3,
    # after 3 moves).
    "lmr": [
        ("tt", DEVICE + ["--features", "tt"]),
        ("tt-lmr", DEVICE + ["--features", "tt-lmr"]),
        ("tt-nmp", DEVICE + ["--features", "tt-nmp"]),
        ("tt-nmp-lmr", DEVICE + ["--features", "tt-nmp-lmr"]),
        ("tt-lmr after 6 moves", DEVICE + ["--features", "tt-lmr", "--lmr-moves", "6"]),
        ("tt-lmr divisor 2.55", DEVICE + ["--features", "tt-lmr", "--lmr-divisor", "255"]),
        ("tt-lmr divisor 1.5", DEVICE + ["--features", "tt-lmr", "--lmr-divisor", "150"]),
        ("tt-lmr from depth 2", DEVICE + ["--features", "tt-lmr", "--lmr-min-depth", "2"]),
    ],
    # Futility pruning, with and without late move reductions,
    # and its settings (default: a margin of 150 per ply, up to 3 plies).
    "fut": [
        ("tt", DEVICE + ["--features", "tt"]),
        ("tt-fut", DEVICE + ["--features", "tt-fut"]),
        ("tt-lmr", DEVICE + ["--features", "tt-lmr"]),
        ("tt-lmr-fut", DEVICE + ["--features", "tt-lmr-fut"]),
        ("tt-lmr-fut margin 100", DEVICE + ["--features", "tt-lmr-fut", "--fut-margin", "100"]),
        ("tt-lmr-fut margin 250", DEVICE + ["--features", "tt-lmr-fut", "--fut-margin", "250"]),
        ("tt-lmr-fut to depth 1", DEVICE + ["--features", "tt-lmr-fut", "--fut-depth", "1"]),
    ],
    # Check extensions against the table alone.
    "ext": [
        ("tt", DEVICE + ["--features", "tt"]),
        ("tt-ext", DEVICE + ["--features", "tt-ext"]),
    ],
    # Killer moves, with and without check extensions.
    "killers": [
        ("tt", DEVICE + ["--features", "tt"]),
        ("tt-killers", DEVICE + ["--features", "tt-killers"]),
        ("tt-ext", DEVICE + ["--features", "tt-ext"]),
        ("tt-ext-killers", DEVICE + ["--features", "tt-ext-killers"]),
    ],
    # SEE and delta pruning in quiescence, on top of killer moves.
    "see": [
        ("tt-killers", DEVICE + ["--features", "tt-killers"]),
        ("tt-killers-see", DEVICE + ["--features", "tt-killers-see"]),
        ("tt-killers-see delta only", DEVICE + ["--features", "tt-killers-see", "--see-prune", "0"]),
        ("tt-killers-see SEE only", DEVICE + ["--features", "tt-killers-see", "--delta-margin", "0"]),
        ("tt-killers-see delta 100", DEVICE + ["--features", "tt-killers-see", "--delta-margin", "100"]),
        ("tt-ext-killers", DEVICE + ["--features", "tt-ext-killers"]),
        ("tt-ext-killers-see", DEVICE + ["--features", "tt-ext-killers-see"]),
    ],
    # Null moves and reductions again, on top of killer moves.
    "retry": [
        ("tt-killers", DEVICE + ["--features", "tt-killers"]),
        ("tt-killers-nmp", DEVICE + ["--features", "tt-killers-nmp"]),
        ("tt-killers-lmr", DEVICE + ["--features", "tt-killers-lmr"]),
        ("tt-ext-killers", DEVICE + ["--features", "tt-ext-killers"]),
        ("tt-ext-killers-nmp", DEVICE + ["--features", "tt-ext-killers-nmp"]),
        ("tt-ext-killers-lmr", DEVICE + ["--features", "tt-ext-killers-lmr"]),
    ],
    # Sliding attacks in the search kernels, engine feature set.
    "attacks": [
        ("hq", DEVICE + ["--features", "tt-killers-see"]),
        ("magic", DEVICE + ["--features", "tt-killers-see", "--attacks", "magic"]),
    ],
    # The evaluations' cost with the engine's feature set.
    "evals-see": [
        ("pesto", DEVICE + ["--features", "tt-killers-see", "--eval", "pesto"]),
        ("terms", DEVICE + ["--features", "tt-killers-see", "--eval", "terms"]),
    ],
    # Checks in quiescence on top of the engine's set, alone and with null moves.
    "qchecks": [
        ("tt-killers-see", DEVICE + ["--features", "tt-killers-see"]),
        ("tt-killers-see-qchecks", DEVICE + ["--features", "tt-killers-see-qchecks"]),
        ("tt-killers-see-qchecks-nmp", DEVICE + ["--features", "tt-killers-see-qchecks-nmp"]),
    ],
    # Principal variation search on top of the engine's set.
    "pvs": [
        ("tt-killers-see", DEVICE + ["--features", "tt-killers-see"]),
        ("tt-killers-see-pvs", DEVICE + ["--features", "tt-killers-see-pvs"]),
    ],
    # Aspiration windows (first margin in centipawns; 0: full windows), engine defaults.
    "aspiration": [
        ("full windows", DEVICE + ["--features", "tt-killers-see"]),
        ("aspiration 15", DEVICE + ["--features", "tt-killers-see", "--aspiration", "15"]),
        ("aspiration 30", DEVICE + ["--features", "tt-killers-see", "--aspiration", "30"]),
        ("aspiration 60", DEVICE + ["--features", "tt-killers-see", "--aspiration", "60"]),
    ],
    # The NNUE eval against terms and PeSTO, engine feature set.
    "nnue": [
        ("terms", DEVICE + ["--features", "tt-killers-see", "--eval", "terms"]),
        ("nnue", DEVICE + ["--features", "tt-killers-see", "--eval", "nnue"]),
        ("pesto", DEVICE + ["--features", "tt-killers-see", "--eval", "pesto"]),
    ],
    # ABDADA on the GPU against the device YBWC, engine feature
    # set; --workers is capped at what fits (1048576: as many as fit).
    "abdada": [
        ("ybwc-device", DEVICE + ["--features", "tt-killers-see"]),
        ("abdada-device 128", ABDADA + ["--workers", "128"]),
        ("abdada-device 512", ABDADA + ["--workers", "512"]),
        ("abdada-device 1024", ABDADA + ["--workers", "1024"]),
        ("abdada-device all, 1 per warp", ABDADA + ["--workers", "1048576"]),
        ("abdada-device 4096, 2 per warp", ABDADA + ["--workers", "4096", "--lanes", "2"]),
        ("abdada-device 8192, 32 per warp", ABDADA + ["--workers", "8192", "--lanes", "32"]),
    ],
    # ABDADA's coordination settings (AbdadaMinDepth, AbdadaDeferFirst) at 1024 workers.
    "abdada-settings": [
        ("ybwc-device", DEVICE + ["--features", "tt-killers-see"]),
        ("abdada from depth 2", ABDADA + ["--workers", "1024"]),
        ("abdada from depth 1", ABDADA + ["--workers", "1024", "--abdada-min-depth", "1"]),
        ("abdada from depth 3", ABDADA + ["--workers", "1024", "--abdada-min-depth", "3"]),
        ("abdada from depth 4", ABDADA + ["--workers", "1024", "--abdada-min-depth", "4"]),
        ("abdada from depth 2, defer first", ABDADA + ["--workers", "1024", "--abdada-defer-first", "1"]),
        ("abdada from depth 1, defer first", ABDADA + ["--workers", "1024", "--abdada-min-depth", "1",
                                              "--abdada-defer-first", "1"]),
        ("abdada from depth 3, defer first", ABDADA + ["--workers", "1024", "--abdada-min-depth", "3",
                                              "--abdada-defer-first", "1"]),
    ],
    # Table size, with the full TT feature set.
    "hash": [
        ("tt 256 MB", DEVICE + ["--features", "tt", "--hash", "256"]),
        ("tt 2048 MB", DEVICE + ["--features", "tt", "--hash", "2048"]),
        ("tt 8192 MB", DEVICE + ["--features", "tt", "--hash", "8192"]),
    ],
}


def run(engine, args, depth, limit):
    # The plain search unless the configuration names a feature set (a later
    # --features wins), whatever the engine's default is.
    cmd = [str(engine), "bench-search", "--depth", str(depth), "--features", "none"] + \
        (["--limit", str(limit)] if limit else []) + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(out.stdout)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    m = re.search(r"TOTAL (\d+) positions \| ([\d.]+) ms \| ([\d,]+) nodes", out.stdout)
    warnings = out.stdout.count("WARNING")
    return float(m.group(2)), int(m.group(3).replace(",", "")), warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(CONFIGS))
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--engine", type=pathlib.Path, default=ENGINE)
    a = ap.parse_args()
    configs = CONFIGS[a.name]
    best = {}
    times = {label: [] for label, _ in configs}
    for p in range(a.passes):
        for label, args in (configs if p % 2 == 0 else list(reversed(configs))):
            ms, nodes, warnings = run(a.engine, args, a.depth, a.limit)
            times[label].append(ms)
            if label not in best or ms < best[label][0]:
                best[label] = (ms, nodes, warnings)
    print(f"{'configuration':36s} {'time to depth ' + str(a.depth):>18s} {'nodes':>16s}")
    for label, _ in configs:
        ms, nodes, warnings = best[label]
        flag = f"  ({warnings} positions short of the depth)" if warnings else ""
        passes = "  (passes: " + ", ".join(f"{x / 1000:.2f}" for x in times[label]) + ")" if a.passes > 2 else ""
        print(f"{label:36s} {ms / 1000:15.2f} s {nodes:16,d}{flag}{passes}")


if __name__ == "__main__":
    main()
