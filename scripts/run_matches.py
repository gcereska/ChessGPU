"""Plays several matches of scripts/run_match.py one after another, for long
unattended runs (start it detached, e.g. with PowerShell Start-Process).

  python scripts/run_matches.py tt nmp [--engine EXE]

Every match plays the same copy of build/Release/chessgpu.exe (or of EXE),
made when the chain starts (build/matches/chain-chessgpu.exe), so later
matches don't pick up builds made while earlier ones play. Each match's console output
goes to build/matches/<match>.out; a line per match (start, end, exit code)
goes to build/matches/chain.log.
"""

import argparse
import datetime
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "matches"


def log(text):
    with open(OUT / "chain.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="+")
    ap.add_argument("--engine", type=pathlib.Path, default=ROOT / "build" / "Release" / "chessgpu.exe")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    engine = OUT / "chain-chessgpu.exe"
    shutil.copy2(a.engine, engine)
    for match in a.matches:
        log(f"start {match}")
        with open(OUT / (match + ".out"), "w", encoding="utf-8") as out:
            code = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_match.py"), match, "--engine",
                                   str(engine)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        log(f"end {match}: exit {code}")


if __name__ == "__main__":
    main()
