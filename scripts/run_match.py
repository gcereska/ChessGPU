"""Engine matches with fastchess (tools/fastchess/fastchess.exe).

  python scripts/run_match.py evals      # PeSTO against material, SPRT
  python scripts/run_match.py terms      # hand-written terms on top of PeSTO against PeSTO, SPRT
  python scripts/run_match.py models     # thread model against warp model, same search, SPRT
  python scripts/run_match.py tt         # transposition table against none, SPRT
  python scripts/run_match.py tt-layouts # packed64 against wide128 table entries, SPRT
  python scripts/run_match.py nmp        # null-move pruning against the table alone, SPRT
  python scripts/run_match.py lmr        # late move reductions against the table alone, SPRT
  python scripts/run_match.py nmp-lmr    # late move reductions with null moves against null moves alone
  python scripts/run_match.py fut        # futility pruning against the table alone, SPRT
  python scripts/run_match.py lmr-fut    # futility pruning with reductions against reductions alone, SPRT
  python scripts/run_match.py ext        # check extensions against the table alone, SPRT
  python scripts/run_match.py killers    # killer moves against the table alone, SPRT
  python scripts/run_match.py ext-killers  # killer moves with check extensions against extensions alone
  python scripts/run_match.py killers-ext  # check extensions with killer moves against killer moves alone
  python scripts/run_match.py killers-see  # SEE and delta pruning on top of killer moves
  python scripts/run_match.py ext-killers-see  # the same with check extensions
  python scripts/run_match.py killers-nmp  # null moves on top of killer moves (also killers-lmr,
                                           # ext-killers-nmp, ext-killers-lmr)
  python scripts/run_match.py see-ext      # check extensions on top of the default (tt-killers-see)
  python scripts/run_match.py see-pvs      # principal variation search on top of the default
  python scripts/run_match.py see-qchecks  # checks in quiescence on top of the default (also see-qchecks-nmp)
  python scripts/run_match.py aspiration   # aspiration windows (30 cp) against full windows
  python scripts/run_match.py aspiration-fl  # the same with extra time after a fail low
  python scripts/run_match.py nnue       # the NNUE evaluation against the hand-written terms, SPRT
  python scripts/run_match.py smoke      # fast games of the default engine against itself:
                                         # no crashes, illegal moves or time losses allowed
  python scripts/run_match.py smoke --summary   # summarize the last run's PGN again

Openings come from tests/search/openings.epd (scripts/make_positions.py
openings), in a seeded random order. SPRT matches play each opening with
both colors; smoke plays each opening once (self-play is deterministic, so
the color-swapped game would repeat the first). Games run one at a time,
since both engines share the GPU and timing must stay fair. SPRT matches
adjudicate like fishtest (resign at 600 cp for 3 moves with both engines
agreeing; draw after move 34 within 20 cp for 8 moves). Smoke games are
played to the end, to exercise every way a game can end. Output (PGN, log,
fastchess state) goes to build/matches/. After the match, a summary lists
how games ended and each engine's depth, time per move and latency.

Both engines run a copy of build/Release/chessgpu.exe made when the match
starts (build/matches/<match>-chessgpu.exe), so the build can change while
a match plays; --engine names another binary to play instead.
"""

import argparse
import collections
import pathlib
import re
import shutil
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FASTCHESS = ROOT / "tools" / "fastchess" / "fastchess.exe"
ENGINE = ROOT / "build" / "Release" / "chessgpu.exe"
OPENINGS = ROOT / "tests" / "search" / "openings.epd"
OUT = ROOT / "build" / "matches"

# The engine's best known search settings (docs/ARCHITECTURE.md, Results).
BASE = {"Search": "ybwc-device", "LeafDepth": "2", "NodeBudget": "2048", "CutWaves": "true"}

MATCHES = {
    "evals": ("pesto", {"Eval": "pesto"}, "material", {"Eval": "material"}, "10+0.1", "sprt"),
    # Hand-written terms on top of PeSTO.
    "terms": ("terms", {"Eval": "terms"}, "pesto", {"Eval": "pesto"}, "10+0.1", "sprt"),
    # Checks in quiescence on top of SEE, alone and with null moves.
    "see-qchecks": ("tt-killers-see-qchecks", {"Features": "tt-killers-see-qchecks"}, "tt-killers-see",
                    {"Features": "tt-killers-see"}, "10+0.1", "sprt"),
    "see-qchecks-nmp": ("tt-killers-see-qchecks-nmp", {"Features": "tt-killers-see-qchecks-nmp"}, "tt-killers-see",
                        {"Features": "tt-killers-see"}, "10+0.1", "sprt"),
    # Principal variation search on top of SEE.
    "see-pvs": ("tt-killers-see-pvs", {"Features": "tt-killers-see-pvs"}, "tt-killers-see",
                {"Features": "tt-killers-see"}, "10+0.1", "sprt"),
    # Aspiration windows against full windows (engine defaults otherwise).
    "aspiration": ("asp30", {"Aspiration": "30"}, "full", {"Aspiration": "0"}, "10+0.1", "sprt"),
    # The same with twice the hard limit after a fail low (FailLowTime 100).
    "aspiration-fl": ("asp30-fl100", {"Aspiration": "30", "FailLowTime": "100"}, "full", {"Aspiration": "0"},
                      "10+0.1", "sprt"),
    # The NNUE eval against the hand-written terms (engine features).
    "nnue": ("nnue", {"Eval": "nnue"}, "terms", {"Eval": "terms"}, "10+0.1", "sprt"),
    # A network from a file (the fourth, trained after the engine copy was made).
    "nnue4": ("nnue4", {"Eval": "nnue", "NnueFile": str(ROOT / "build" / "nnue" / "net_v4.nnue")}, "terms",
              {"Eval": "terms"}, "10+0.1", "sprt"),
    # The fifth network (8.4M positions), from a file as well.
    "nnue5": ("nnue5", {"Eval": "nnue", "NnueFile": str(ROOT / "build" / "nnue" / "net_v5.nnue")}, "terms",
              {"Eval": "terms"}, "10+0.1", "sprt"),
    "models": ("thread", {"Model": "thread"}, "warp", {"Model": "warp", "BlockSize": "64"}, "10+0.1", "sprt"),
    "smoke": ("a", {}, "b", {}, "2+0.02", "smoke"),
    # The transposition table against the engine without one,
    # and its two entry layouts against each other.
    "tt": ("tt", {"Features": "tt"}, "none", {"Features": "none"}, "10+0.1", "sprt"),
    "tt-layouts": ("packed64", {"Features": "tt", "TTLayout": "packed64"},
                   "wide128", {"Features": "tt", "TTLayout": "wide128"}, "10+0.1", "sprt"),
    "nmp": ("tt-nmp", {"Features": "tt-nmp"}, "tt", {"Features": "tt"}, "10+0.1", "sprt"),
    "lmr": ("tt-lmr", {"Features": "tt-lmr"}, "tt", {"Features": "tt"}, "10+0.1", "sprt"),
    "nmp-lmr": ("tt-nmp-lmr", {"Features": "tt-nmp-lmr"}, "tt-nmp", {"Features": "tt-nmp"}, "10+0.1", "sprt"),
    "fut": ("tt-fut", {"Features": "tt-fut"}, "tt", {"Features": "tt"}, "10+0.1", "sprt"),
    "lmr-fut": ("tt-lmr-fut", {"Features": "tt-lmr-fut"}, "tt-lmr", {"Features": "tt-lmr"}, "10+0.1", "sprt"),
    "ext": ("tt-ext", {"Features": "tt-ext"}, "tt", {"Features": "tt"}, "10+0.1", "sprt"),
    "killers": ("tt-killers", {"Features": "tt-killers"}, "tt", {"Features": "tt"}, "10+0.1", "sprt"),
    "ext-killers": ("tt-ext-killers", {"Features": "tt-ext-killers"}, "tt-ext", {"Features": "tt-ext"}, "10+0.1",
                    "sprt"),
    "killers-ext": ("tt-ext-killers", {"Features": "tt-ext-killers"}, "tt-killers", {"Features": "tt-killers"},
                    "10+0.1", "sprt"),
    "killers-see": ("tt-killers-see", {"Features": "tt-killers-see"}, "tt-killers", {"Features": "tt-killers"},
                    "10+0.1", "sprt"),
    "ext-killers-see": ("tt-ext-killers-see", {"Features": "tt-ext-killers-see"}, "tt-ext-killers",
                        {"Features": "tt-ext-killers"}, "10+0.1", "sprt"),
    # On top of SEE (the default since killers-see won): check extensions.
    "see-ext": ("tt-ext-killers-see", {"Features": "tt-ext-killers-see"}, "tt-killers-see",
                {"Features": "tt-killers-see"}, "10+0.1", "sprt"),
    "killers-nmp": ("tt-killers-nmp", {"Features": "tt-killers-nmp"}, "tt-killers", {"Features": "tt-killers"},
                    "10+0.1", "sprt"),
    "killers-lmr": ("tt-killers-lmr", {"Features": "tt-killers-lmr"}, "tt-killers", {"Features": "tt-killers"},
                    "10+0.1", "sprt"),
    "ext-killers-nmp": ("tt-ext-killers-nmp", {"Features": "tt-ext-killers-nmp"}, "tt-ext-killers",
                        {"Features": "tt-ext-killers"}, "10+0.1", "sprt"),
    "ext-killers-lmr": ("tt-ext-killers-lmr", {"Features": "tt-ext-killers-lmr"}, "tt-ext-killers",
                        {"Features": "tt-ext-killers"}, "10+0.1", "sprt"),
}

# PGN Termination values that mean a game ended normally.
CLEAN_TERMINATIONS = {"normal", "adjudication"}


def engine_args(engine, name, options):
    # "NnueFile" is a command-line option of the engine (chessgpu uci
    # --nnue-file PATH: weights loaded before anything runs), not a UCI one.
    options = dict(options)
    nnue_file = options.pop("NnueFile", None)
    uci = f"uci --nnue-file {nnue_file}" if nnue_file else "uci"
    args = ["-engine", f"cmd={engine}", f"name={name}", f"args={uci}"]
    for k, v in {**BASE, **options}.items():
        args.append(f"option.{k}={v}")
    return args


def read_games(pgn):
    """Returns [(tags, movetext)] for every game in a PGN file."""
    games, tags, text = [], {}, []
    for line in pgn.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\[(\w+) "(.*)"\]$', line)
        if m:
            if text:
                games.append((tags, " ".join(text)))
                tags, text = {}, []
            tags[m.group(1)] = m.group(2)
        elif line.strip():
            text.append(line.strip())
    if tags:
        games.append((tags, " ".join(text)))
    return games


def summarize(pgn):
    """Prints how games ended and per-engine move statistics; returns the
    number of games that didn't end cleanly (time loss, illegal move, crash)."""
    games = read_games(pgn)
    terminations = collections.Counter()
    reasons = collections.Counter()
    bad = []
    depth = collections.defaultdict(list)
    time_s = collections.defaultdict(list)
    latency = collections.defaultdict(list)
    left = collections.defaultdict(list)
    for tags, text in games:
        termination = tags.get("Termination", "?")
        terminations[termination] += 1
        comments = re.findall(r"\{([^}]*)\}", text)
        last = comments[-1].split(", ")[-1] if comments else ""
        reason = last if last and not re.match(r"^([-+]?M?[\d.]+/\d+|book|\w+=)", last) else "?"
        reasons[reason] += 1
        if termination not in CLEAN_TERMINATIONS:
            bad.append(f"round {tags.get('Round', '?')}: {tags.get('White')} vs {tags.get('Black')}: "
                       f"{termination}, {reason}")
        side = 0 if tags.get("FEN", "w").split()[1:2] in ([], ["w"]) else 1
        names = (tags.get("White"), tags.get("Black"))
        for c in comments:
            m = re.match(r"^([-+]?M?[\d.]+)/(\d+) ([\d.]+)s", c)
            if m:
                name = names[side]
                depth[name].append(int(m.group(2)))
                time_s[name].append(float(m.group(3)))
                for key, store in (("latency", latency), ("tl", left)):
                    v = re.search(rf"\b{key}=(-?[\d.]+)s", c)
                    if v:
                        store[name].append(float(v.group(1)))
            side ^= 1
    print(f"\n{len(games)} games in {pgn}")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.most_common()))
    print("endings:      " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    for name in sorted(depth):
        d, t = depth[name], time_s[name]
        line = (f"{name}: {len(d)} moves | depth mean {statistics.mean(d):.2f}, median {statistics.median(d):g}, "
                f"min {min(d)}, max {max(d)} | time per move mean {1000 * statistics.mean(t):.0f} ms, "
                f"max {1000 * max(t):.0f} ms")
        if latency[name]:
            line += f" | latency max {1000 * max(latency[name]):.0f} ms"
        if left[name]:
            line += f" | least time left {1000 * min(left[name]):.0f} ms"
        print(line)
    for b in bad:
        print("NOT CLEAN: " + b)
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match", choices=sorted(MATCHES))
    ap.add_argument("--rounds", type=int, default=0,
                    help="rounds (default: SPRT up to 500 opening pairs, smoke every opening once)")
    ap.add_argument("--seed", type=int, default=1, help="opening order seed")
    ap.add_argument("--summary", action="store_true", help="only summarize the last run's PGN")
    ap.add_argument("--engine", type=pathlib.Path, help="engine binary to play (default: a copy of this build's)")
    a = ap.parse_args()
    name1, opt1, name2, opt2, tc, kind = MATCHES[a.match]
    pgn = OUT / (a.match + ".pgn")
    if a.summary:
        sys.exit(1 if summarize(pgn) else 0)
    if not FASTCHESS.exists():
        sys.exit(f"fastchess not found at {FASTCHESS}")
    OUT.mkdir(parents=True, exist_ok=True)
    # fastchess gets the FENs alone (the comments would read as EPD operations).
    openings = OUT / "openings.epd"
    fens = [line.split(";")[0].strip() for line in OPENINGS.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    openings.write_text("\n".join(fens) + "\n")
    rounds = a.rounds or (500 if kind == "sprt" else len(fens))
    engine = a.engine
    if engine is None:
        engine = OUT / (a.match + "-chessgpu.exe")
        shutil.copy2(ENGINE, engine)
    cmd = [str(FASTCHESS)] + engine_args(engine, name1, opt1) + engine_args(engine, name2, opt2)
    cmd += ["-each", f"tc={tc}", "proto=uci",
            "-openings", f"file={openings}", "format=epd", "order=random", "-srand", str(a.seed),
            "-rounds", str(rounds), "-games", "2" if kind == "sprt" else "1", "-concurrency", "1", "-recover",
            "-pgnout", f"file={pgn}", "append=false", "nodes=true", "nps=true", "timeleft=true", "latency=true",
            "-log", f"file={OUT / (a.match + '.log')}", "level=warn", "append=false"]
    if kind == "sprt":
        cmd += ["-sprt", "elo0=0", "elo1=10", "alpha=0.05", "beta=0.05",
                "-resign", "movecount=3", "score=600", "twosided=true",
                "-draw", "movenumber=34", "movecount=8", "score=20"]
    print(" ".join(cmd), flush=True)
    # In OUT, where fastchess also autosaves its state (config.json).
    code = subprocess.run(cmd, cwd=OUT).returncode
    bad = summarize(pgn) if pgn.exists() else 0
    sys.exit(code or (1 if bad else 0))


if __name__ == "__main__":
    main()
