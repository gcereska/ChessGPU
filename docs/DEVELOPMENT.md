# Development

How to build each configuration, run the tests, check that the tests catch bugs, benchmark a change and settle it with a match. The conventions the code follows are at the end. Why the tests are built the way they are is in [ARCHITECTURE.md](ARCHITECTURE.md#testing).

Contents:
[Builds](#builds) ·
[Tests](#tests) ·
[Planted bugs](#planted-bugs) ·
[Benchmarks](#benchmarks) ·
[Matches](#matches) ·
[Profiling](#profiling) ·
[Other scripts](#other-scripts) ·
[Conventions](#conventions)

## Builds

| Preset | Tree | Configuration |
|---|---|---|
| `release` | `build/` | optimized: the engine as used |
| `debug` | `build/` | host code unoptimized, with assertions; device code still optimized, since debugging happens on the CPU (see [Conventions](#conventions)) |
| `profile` | `build/` | Release plus source line information for Nsight Compute (`-lineinfo`) and ptxas's register and spill report |
| `dev` | `build/dev` | Release with a reduced set of variants, for quick iterations on kernels |

```bash
cmake --preset default              # configures build/ (Visual Studio 2026, CUDA)
cmake --build --preset release      # or: debug, profile
cmake --preset dev && cmake --build --preset dev
```

Every variant is a kernel instantiation of its own, so a full build compiles a lot of device code. The search kernels of each feature set and table layout are generated into separate units (from `src/gpu/kernels.cu.in`) that compile in parallel. A change to host-only code (`src/host`, `src/engine`, the CPU tests) relinks quickly; a change to a header the kernels include rebuilds every kernel unit. [ENGINE.md](ENGINE.md#building) lists the CMake options that choose the variants.

Build notes:
- Visual Studio's CUDA dependency scanner stops at `<cuda/atomic>`, so project headers are included before it.
- CMake caches the `CHESSGPU_*` variant lists: changing one needs `-D` when reconfiguring.

## Tests

```bash
ctest --preset release               # everything
ctest --preset release -L unit       # one label
```

| Label | Runs | Needs |
|---|---|---|
| `unit` | `unit_tests`: rules, move generation, evaluation, the CPU search against a reference, every search technique, MultiPV, ABDADA, the NNUE and its trainer, the tuner, self-play | the CPU |
| `gpu` | `gpu_tests` (device code against the CPU, GPU perft, the GPU searches) and a smoke test | the GPU |
| `engine` | `engine_tests`: the UCI protocol, time management and every search backend | the GPU |
| `perft` | the perft suites through `chessgpu perft-suite`, for every compiled board and model and on the CPU | the GPU |
| `sanitizer` | `gpu_tests --quick` under compute-sanitizer's memcheck, racecheck, synccheck and initcheck | compute-sanitizer |
| `gui` | the GUI's tests, with this build's engine | the `.venv` with PySide6 |

Each test executable takes a filter and runs only the tests whose names contain it:

```bash
build/tests/Release/unit_tests.exe killers
build/tests/Release/gpu_tests.exe --quick see       # --quick: smaller workloads, as under the sanitizers
build/tests/Release/engine_tests.exe multipv
```

**The GUI's tests** run headless, on Qt's offscreen platform:

```bash
cd gui
../.venv/Scripts/python -m pytest -q          # -m slow adds deeper perft checks
```

With `CHESSGPU_ENGINE` set to an engine executable, they include the tests that play and review games with the real engine; without it, those are skipped. Most use `tests/fake_engine.py` instead, a scripted UCI engine that misbehaves on request: a slow start, a crash, an illegal move, no move, a search that never ends.

New code is tested on the CPU first, then on the GPU, then under compute-sanitizer.

## Planted bugs

`scripts/planted_bugs.py` checks that the tests catch bugs. A campaign plants deliberate bugs one at a time (a wrong comparison, a missing line, swapped arguments), rebuilds a copy of the sources, runs the tests that should fail, and logs whether they did. A bug the tests miss means a test is missing, unless the bug provably can't change any result.

```bash
python scripts/planted_bugs.py multipv-cpu                 # one campaign
python scripts/planted_bugs.py gui-analysis --only "pass"  # only the bugs whose title contains "pass"
```

| Campaign | Covers |
|---|---|
| `killers-cpu`, `see-cpu`, `pvs-cpu`, `qchecks-cpu`, `aspiration-cpu`, `multipv-cpu` | search techniques, on the CPU side |
| `abdada-cpu`, `move32-cpu`, `nnue-cpu` | ABDADA's shared code, 32-bit moves, the NNUE and its trainer |
| `gpu-explicit`, `gpu-abdada` | code that only runs on the GPU: killers and SEE in the YBWC coordinator, ABDADA's kernel (needs the GPU) |
| `gui-look`, `gui-analysis`, `gui-events` | the GUI: the look and feel; the game tree, PGN, analysis and reviews; tournaments and databases |

Each campaign works on a copy of the sources, in a build tree of its own under `build/mut-*`, so the working tree can be edited and built meanwhile. Logs go to `build/planted_bugs/<campaign>.log`.

## Benchmarks

GPU timings vary from run to run, so comparisons need care:
- give every build compared the same `--split` (the automatic split differs between execution models);
- keep the best of several runs;
- alternate the order of the configurations, in separate processes (A-B-B-A).

| Tool | Measures |
|---|---|
| `chessgpu bench [--all] [--repeat N]` | GPU perft speed on six positions, for each compiled board and model |
| `chessgpu bench-search --depth 8` | time to depth on the 50 positions of `tests/search/suite.epd` |
| `chessgpu bench-tasks` | the search kernels alone: throughput on many tasks, and one worker's speed on one task |
| `python scripts/bench_variants.py <set> --passes 3` | `bench-search` for a named set of configurations, in alternating order, keeping each one's best total |

`bench_variants.py` has a set for most questions the project has asked: `models`, `tt`, `killers`, `see`, `nmp`, `lmr`, `evals-see`, `abdada`, `nnue`, `pvs`, `qchecks` and more. `--engine EXE` benchmarks a saved build instead of the current one.

## Matches

A technique becomes the default only by winning a match against the best version so far.

```bash
python scripts/run_match.py killers-see             # one named match (the script lists them)
python scripts/run_match.py smoke                   # fast self-play: no crash, illegal move or time loss allowed
python scripts/run_matches.py see-pvs see-qchecks   # several matches in a row
```

- **Runner:** [fastchess](https://github.com/Disservin/fastchess), expected at `tools/fastchess/fastchess.exe` (not included).
- **Format:** 10 seconds plus 0.1 per move, the 200 balanced openings of `tests/search/openings.epd` with both colors, one game at a time (both engines share the GPU).
- **Statistics:** an SPRT with elo0 = 0, elo1 = 10 and alpha = beta = 0.05. A match may be stopped early once both 0 and +10 Elo lie outside the error bar.
- **Adjudication,** as fishtest's: a win at 600 centipawns for 3 moves with both engines agreeing; a draw after move 34 within 20 centipawns for 8 moves.
- **Isolation:** each match plays a copy of the engine made when it starts, so building during a match is safe. Keep the CPU otherwise idle, though: a busy CPU slows the GPU engine's searches and can cost it games on time. Long chains of matches are best started as a detached process.

PGNs, logs and fastchess's state go to `build/matches/`. After a match, a summary lists how the games ended and each engine's depth, time per move and latency (`--summary` prints it again).

## Profiling

- **Nsight Compute** on the Profile build: the kernels carry source lines (`-lineinfo`), and the build prints ptxas's register and spill counts for every kernel.
- **NVTX ranges** mark the phases of GPU work for Nsight Systems: perft's CPU split, upload, kernel and download, and the YBWC search.
- **Debugging happens on the CPU:** the search is plain C++ that runs unchanged on the host, so a debugger and the CPU tests reach exactly the code the GPU runs.

## Other scripts

| Script | Does |
|---|---|
| `stockfish_perft.py` | perft counts from Stockfish: `verify` a suite file, print `counts` for a FEN, or `divide` |
| `compare_divide.py` | finds the first position where the engine's legal moves differ from Stockfish's |
| `make_positions.py` | generates the benchmark suite and the match openings by seeded self-play |
| `make_tuning_set.py` | turns PGN games into positions, for the tuner or for labeling |

Stockfish is expected at `tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe`, or wherever `STOCKFISH` points; it isn't included.

## Conventions

- **Shared CPU and GPU code.** The code in `src/core/` is header-only and marked `CB_HD` (`__host__ __device__`). Calling host-only code from it is a compile error, not a warning.
- **No runtime indexing into board fields.** Registers can't be indexed by a runtime value, so CUDA would move the board to local memory. Piece types are template parameters, and runtime choices use selects or masked XORs. Board classes implement only primitives; make and unmake are generic.
- **Variants are template parameters,** and the host picks one at run time. A new design choice becomes a variant with an option of its own, rather than replacing what was there.
- **Variants that lose stay in the code.** Winning a match makes a variant the default; the others remain available for later combinations.
- **Tunable numbers are run-time settings** (`PruneParams`), passed to every kernel, so one compiled search serves every setting.
- **Defaults come from measurements.** A change is first compared by time to depth, then played as a match, and the results go into [ARCHITECTURE.md](ARCHITECTURE.md#results).
