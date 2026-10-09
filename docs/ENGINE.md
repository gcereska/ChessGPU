# The engine

`chessgpu.exe` is a console program. Started without arguments it is a [UCI](https://www.chessprogramming.org/UCI) engine, for a chess GUI or a match runner to drive; started with a command, it is a set of tools for perft, single searches, benchmarks and training data. This page covers building it, playing with it, its options and its commands. How it works is in [ARCHITECTURE.md](ARCHITECTURE.md).

Contents:
[Building](#building) ·
[Playing with it](#playing-with-it) ·
[UCI options](#uci-options) ·
[Feature sets](#feature-sets) ·
[Command-line tools](#command-line-tools) ·
[The NNUE and its training](#the-nnue-and-its-training)

## Building

**Requirements:**
- Windows with Visual Studio 2026 (the CMake presets use its generator);
- CUDA Toolkit 13.2 and CMake 3.28 or newer;
- an NVIDIA GPU of compute capability 7.5 (Turing) or newer, which CUDA 13 requires. CMake builds for the GPU in the machine; to build for another GPU, or without one, set the architecture: `cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=89`.

It has been tested on Windows 11 with a compute capability 12.0 GPU ([Hardware](ARCHITECTURE.md#hardware)), where its defaults were tuned. On a GPU with less memory, lower `Hash` (2 GB by default).

```bash
cmake --preset default
cmake --build --preset release
```

This builds `build/Release/chessgpu.exe`. The `debug` and `profile` presets build the other configurations ([DEVELOPMENT.md](DEVELOPMENT.md#builds)).

**Variants and build time.** Every variant of the search (boards, execution models, evaluations, pruning feature sets and more) is compiled into the one executable, so a full build takes a long time and the binary is large. The `dev` preset builds a reduced set into `build/dev`. These CMake lists choose the variants by hand; each takes a semicolon-separated list and defaults to everything:

| Option | Values |
|---|---|
| `CHESSGPU_BOARDS` | `quad`, `std` |
| `CHESSGPU_MODELS` | `thread`, `warp` |
| `CHESSGPU_EVALS` | `material`, `pesto`, `terms`, `nnue` |
| `CHESSGPU_ORDERINGS` | `staged`, `list` |
| `CHESSGPU_STACKS` | `local`, `shared` |
| `CHESSGPU_FEATURES` | the [feature sets](#feature-sets), or `all` |
| `CHESSGPU_TT_LAYOUTS` | `packed64`, `wide128` |
| `CHESSGPU_SEARCH_ATTACKS` | `hq`, `magic` (sliding-piece attacks in the search kernels) |
| `CHESSGPU_ATTACKS` | `hq`, `magic`, `kogge` (in the perft kernels) |
| `CHESSGPU_MOVES` | `16`, `32` (move encodings in perft) |
| `CHESSGPU_DEVICE_SEARCHES` | `ybwc`, `abdada` |

For example: `cmake --preset default -DCHESSGPU_EVALS=nnue -DCHESSGPU_MODELS=thread`. Asking at run time for a variant that isn't compiled in is an error, not a silent fallback. CMake caches these lists, so changing one later needs `-D` again.

## Playing with it

```bash
build/Release/chessgpu.exe          # the same as: chessgpu.exe uci
```

It has been used with the GUI in `gui/` ([GUI.md](GUI.md)) and with the match runner [fastchess](https://github.com/Disservin/fastchess). Other UCI GUIs should work the same way: point them at the executable.

A session, after `uci` and `isready` (`>` marks what was sent; long lines are cut):

```text
> position startpos moves e2e4 e7e5 g1f3
> go wtime 60000 btime 60000 winc 1000 binc 1000
info string search ybwc-device gpu quad/nnue/thread/staged/local block 128 tt-killers-see wide128 …
info depth 1 score cp -22 nodes 31 nps 1091 time 28 pv d7d6
info depth 2 score cp -7 nodes 170 nps 5612 time 30 pv d7d6
info depth 3 score cp -21 nodes 1596 nps 46891 time 34 pv b8c6 d2d4
info depth 4 score cp -11 nodes 9593 nps 220045 time 43 pv b8c6 b1c3 f8e7
info depth 5 score cp -14 nodes 42478 nps 625170 time 67 pv b8c6 b1c3 g8f6 a2a3
info depth 6 score cp -10 nodes 166732 nps 1269728 time 131 pv b8c6 b1c3 g8f6 a2a3 a7a6
info depth 7 score cp -19 nodes 959205 nps 4191120 time 228 pv b8c6 d2d4 e5d4 f3d4 c6d4 d1d4
info depth 8 score cp -15 nodes 3846278 nps 9763355 time 393 pv b8c6 d2d4 e5d4 f3d4 c6d4 d1d4 g8f6
info depth 9 score cp -39 nodes 54946803 nps 44726472 time 1228 pv b8c6 d2d4 e5d4 f3d4 c6d4 d1d4 g8f6 b1c3
info depth 10 score cp -26 nodes 180083761 nps 67534838 time 2666 pv g8f6 b1c3 b8c6 d2d4 f8b4 d4d5 f6e4 …
bestmove g8f6
```

The speed (`nps`) climbs from about a thousand nodes per second at depth 1 to tens of millions by depth 10. Small trees can't keep the GPU busy ([ARCHITECTURE.md](ARCHITECTURE.md#future-work)).

**What to expect:**
- *Depth:* at 10 seconds plus 0.1 per move it reaches about depth 8 to 10.
- *Start-up:* the GPU search and its 2 GB transposition table are set up when the engine is first asked `isready`, which takes a moment.
- *One game at a time:* GPU engines in simultaneous games share the GPU and slow each other down, so play matches one game at a time.

**Protocol details:**
- *Commands:* `uci`, `isready`, `setoption`, `ucinewgame`, `position`, `go` (with `wtime`, `btime`, `winc`, `binc`, `movestogo`, `depth`, `nodes`, `movetime` or `infinite`), `stop` and `quit`. `go ponder` is searched like a plain `go`, and `searchmoves` isn't supported. Options can't change during a search.
- *Output:* an `info depth … score … nodes … nps … time … pv …` line after every finished iteration; with `MultiPV` above 1, one such line per reported move, tagged `multipv 1`, `multipv 2` and so on. One `info string` line names the search that runs.
- *Another network:* `chessgpu.exe uci --nnue-file PATH` evaluates with NNUE weights from a file instead of the network compiled in ([The NNUE and its training](#the-nnue-and-its-training)).

**Time management.** With a clock, the engine aims to spend the time left (minus `MoveOverhead`) divided by 30, or by `movestogo`, plus 80% of the increment. It starts no new iteration after 0.6 times that target and stops a running one at 3 times it, never using more than 80% of the clock. Depth 1 always finishes, so there is always a move. With `movetime` it stops at that time. With `depth`, it stops after that depth; with `nodes`, it starts no new iteration once the count is passed.

## UCI options

The defaults are the configuration that won its matches; the rest is there to compare against. Options that matter to only one search or one feature set say so.

**Search and resources**

| Option | Type | Default | Meaning |
|---|---|---|---|
| `Search` | combo | `ybwc-device` | the search: `ybwc-device` (YBWC coordinated on the GPU), `ybwc-host` (YBWC coordinated by the CPU), `gpu-independent` (independent subtrees), `abdada-device`, `abdada-cpu`, or `cpu` (one CPU core) |
| `Hash` | spin | 2048 | transposition table size in MB (1 to 65536) |
| `Clear Hash` | button | | empties the transposition table |
| `MultiPV` | spin | 1 | how many lines to report (1 to 256): the best N moves, each with its score and line |
| `MoveOverhead` | spin | 50 | milliseconds kept back from every move for communication delays |
| `Threads` | spin | 4 | `abdada-cpu`: worker threads |
| `Workers` | spin | 2048 | `abdada-device`: GPU workers (capped at what fits) |

**Variants**

| Option | Default | Values |
|---|---|---|
| `Board` | `quad` | `quad` (4 bitboards), `std` (8 bitboards) |
| `Model` | `thread` | `thread` (one search per GPU thread), `warp` (one search per warp) |
| `Eval` | `nnue` | `nnue`, `terms` (PeSTO's tables plus hand-written terms), `pesto`, `material` |
| `Ordering` | `staged` | `staged` (moves generated lazily, in stages), `list` (all generated, then picked) |
| `Stack` | `local` | `local`, `shared`: where the thread model keeps its search stack |
| `Features` | `tt-killers-see` | a [feature set](#feature-sets) |
| `Attacks` | `hq` | `hq` (hyperbola quintessence), `magic` (magic bitboards): sliding-piece attacks in the GPU search |
| `TTLayout` | `wide128` | `wide128` (2 entries per bucket, full keys), `packed64` (4 per bucket, 16-bit checks) |

**YBWC** (the `ybwc-device` and `ybwc-host` searches)

| Option | Default | Meaning |
|---|---|---|
| `LeafDepth` | 2 | with this many plies left or fewer, a subtree is one GPU task |
| `NodeBudget` | 2048 | a task that searches more nodes than this is split (0: never) |
| `CutWaves` | true | at nodes expected to cut off, start the remaining moves in waves of 1, 2, 4, … rather than all at once |
| `ExplicitPlies` | 8 | at most this many plies of coordinated nodes |
| `Lanes` | 1 | threads per warp that search (`ybwc-device`, `abdada-device`) |
| `BlockSize` | 128 | GPU threads per block |
| `SplitDepth` | 2 | `gpu-independent`: plies the CPU expands before the GPU searches the subtrees |

**ABDADA** (the `abdada-cpu` and `abdada-device` searches): `AbdadaMinDepth` (default 2) is the fewest plies left at which moves are claimed and deferred, and `AbdadaDeferFirst` (default false) defers a node's first move too while another worker searches it.

**Pruning settings.** These tune techniques that only some feature sets contain; with the default set, only the last two have an effect.

| Option | Default | Meaning |
|---|---|---|
| `NullMinDepth` | 2 | null moves: the fewest plies left to try one |
| `NullReduction`, `NullDivisor` | 3, 6 | null moves: reduce by NullReduction + depth / NullDivisor plies |
| `LmrMinDepth` | 3 | late move reductions: the fewest plies left to reduce |
| `LmrMoves` | 3 | late move reductions: how many moves are searched first without reduction |
| `LmrBase`, `LmrDivisor` | 75, 225 | late move reductions: reduce by LmrBase / 100 + ln(depth) · ln(move number) / (LmrDivisor / 100) plies |
| `FutDepth`, `FutMargin` | 3, 150 | futility pruning: the most plies left, and the margin per ply in centipawns |
| `DeltaMargin` | 200 | delta pruning in quiescence: the margin in centipawns (0: off) |
| `SeePrune` | true | in quiescence, skip captures that lose material by static exchange evaluation |

**Time**

| Option | Default | Meaning |
|---|---|---|
| `Aspiration` | 0 | aspiration windows from depth 4: the first margin in centipawns (0: full windows) |
| `FailLowTime` | 0 | after an aspiration window fails low, let the iteration run this much longer, in % of the hard limit (0 to 400) |

## Feature sets

A feature set is a combination of search techniques, compiled as a variant of its own and chosen with `Features` (or `--features` on the command line). In this table, "the table" is the transposition table used for move ordering, for cutoffs and in quiescence, unless the row says otherwise.

| Feature set | Techniques |
|---|---|
| `none` | plain alpha-beta with quiescence search |
| `tt` | the table |
| `tt-noqs` | the table, except in quiescence |
| `tt-order` | the table for move ordering only, without cutoffs |
| `tt-killers` | the table and killer moves |
| **`tt-killers-see`** | **the default:** the table, killer moves, and SEE and delta pruning in quiescence |
| `tt-nmp`, `tt-lmr`, `tt-fut`, `tt-ext` | the table plus null-move pruning, late move reductions, futility pruning or check extensions |
| `tt-nmp-lmr`, `tt-lmr-fut` | the table plus two of those |
| `tt-ext-killers`, `tt-ext-killers-see` | `tt-killers` and `tt-killers-see` with check extensions |
| `tt-killers-nmp`, `tt-killers-lmr`, `tt-ext-killers-nmp`, `tt-ext-killers-lmr` | killer moves (and check extensions) with null moves or reductions |
| `tt-killers-see-pvs` | the default plus principal variation search |
| `tt-killers-see-qchecks` | the default plus quiet checking moves at quiescence's first ply |
| `tt-killers-see-qchecks-nmp` | the same plus null-move pruning |

[ARCHITECTURE.md](ARCHITECTURE.md#pruning) has each technique's measurements and match result.

## Command-line tools

`chessgpu.exe help` lists every command and option. The commands:

| Command | Does |
|---|---|
| `search` | searches one position, printing UCI info lines |
| `perft <depth>` | counts the leaves of the move tree (`--divide`: per root move) |
| `perft-suite` | checks perft counts against a suite file (default `tests/perft/standard.epd`) |
| `bench` | GPU perft speed on six positions (`--all`: every compiled board and model) |
| `bench-search` | time to depth on the 50 positions of `tests/search/suite.epd` |
| `bench-tasks` | the GPU search kernels alone: throughput, and one worker's speed |
| `selfplay` | CPU self-play games, written out as scored positions |
| `label` | adds a search score to every position in a file |
| `train-nnue` | trains the NNUE on scored positions |
| `tune` | Texel-tunes the hand-written evaluation terms |
| `info` | prints the GPU's properties |

**Examples:**

```bash
chessgpu.exe search --depth 10                                  # the start position, default search
chessgpu.exe search --fen "<FEN>" --movetime 2000               # a position, for two seconds
chessgpu.exe search --search cpu --features tt --depth 8        # the CPU search with another feature set
chessgpu.exe perft 6 --divide                                   # 119,060,324 leaves from the start
chessgpu.exe perft 5 --model cpu --board std                    # perft on the CPU, with the other board
chessgpu.exe bench --all --repeat 3                             # GPU perft, every compiled variant
chessgpu.exe bench-search --depth 8 --eval terms                # time to depth with another evaluation
```

**Options** shared by the commands (`help` has the full list):
- *the position:* `--fen "<FEN>"` and `--moves "<UCI moves>"`;
- *variants:* `--search`, `--board`, `--model`, `--eval`, `--features`, `--ordering`, `--stack`, `--attacks`, `--tt-layout`, and `--move-bits 16|32` for perft;
- *settings:* the UCI options above in command-line form, such as `--hash`, `--leaf-depth`, `--node-budget`, `--cut-waves` and `--no-cut-waves`, `--lmr-base`, `--delta-margin`;
- *limits:* `--depth`, `--movetime` and `--nodes`.

## The NNUE and its training

The default evaluation is a small neural network: 768 inputs, 2 × 32 hidden neurons, integer arithmetic ([ARCHITECTURE.md](ARCHITECTURE.md#evaluation) describes it). Its weights are compiled in from `src/host/nnue_default.cpp`, and the executable can train new ones:

```bash
chessgpu.exe selfplay --games 20000 --depth 6 --threads 12 --file selfplay.txt   # games, as scored positions
chessgpu.exe train-nnue --file selfplay.txt --out net.nnue                       # train a network
chessgpu.exe uci --nnue-file net.nnue                                            # play with it
```

`train-nnue` always writes the weights file (`--out`); adding `--header src/host/nnue_default.cpp` also writes them as C++ source, compiled in at the next build.

- **Data:** one position per line, `<FEN> | <result> | <game> | <score>`: the game's result for White (1, 0.5 or 0), the game's number, and the search's score in centipawns from White's side. `selfplay` writes these lines. `label --file IN --out OUT --depth D` adds scores to lines that have only the first three fields, such as those `scripts/make_tuning_set.py` makes from PGN games.
- **Training:** the network learns 0.75 × the score's expected result (a logistic curve, 400 centipawns per factor of ten) plus 0.25 × the game's result; `--lambda` sets that weight. Every fifth game is held out, and the trainer prints the held-out loss after every epoch.
