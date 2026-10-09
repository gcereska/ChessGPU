// ChessGPU command line: the UCI engine (the default) and tools for perft,
// searching single positions and benchmarks.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/attacks_kogge.h"
#include "core/board.h"
#include "engine/backends.h"
#include "engine/uci.h"
#include "gpu/device_context.h"
#include "gpu/perft_gpu.h"
#include "host/epd.h"
#include "host/fen.h"
#include "host/game_history.h"
#include "host/host_tables.h"
#include "host/perft_cpu.h"
#include "host/position_moves.h"
#include "host/profiling.h"
#include "host/search_backend.h"
#include "host/task_runner.h"
#include "host/nnue_train.h"
#include "host/selfplay.h"
#include "host/tune.h"
#include "host/uci_move.h"

using namespace cb;

namespace {

// The help text; {EVAL}, {FEATURES} and {FEATURE_LIST} are filled in by
// usage() from the build itself, so they can't go stale.
constexpr const char* kUsage = R"(usage:
  chessgpu [uci]                 run as a UCI engine (the default without arguments)
  chessgpu search [options]      search one position, printing UCI info lines
  chessgpu info                  the GPU's properties
  chessgpu perft <depth> [options]
                                 count the move tree's leaves (--divide: per root move)
  chessgpu perft-suite [--file PATH] [--max-nodes N] [options]
                                 check perft counts against a suite (default: tests/perft/standard.epd)
  chessgpu bench [--repeat N] [--all] [options]
                                 GPU perft speed on the benchmark positions
  chessgpu bench-search [--file PATH] [--depth D] [--limit N] [options]
                                 time to depth D (default 8) on each position of a suite
                                 (default: tests/search/suite.epd), with the chosen search
  chessgpu bench-tasks [--depth D] [--split S] [--node-budget N] [--repeat N] [--all] [options]
                                 GPU search kernels alone: throughput on many tasks (each stopped
                                 after N nodes with a budget), and the speed of one worker on one task
  chessgpu selfplay [--games N] [--depth D] [--threads T] [--random-plies R] [--seed S] [--file PATH] [options]
                                 CPU self-play games, written as scored positions (training data)
  chessgpu label --file IN --out OUT [--depth D] [--threads T] [options]
                                 add each position's search score
  chessgpu train-nnue [--file PATH] [--epochs N] [--lambda L] [--lr X] [--batch N] [--threads T] [--seed S]
                      [--hidden N] [--weight-decay X] [--out PATH] [--header PATH]
                                 train the NNUE on scored self-play positions
  chessgpu tune [--file PATH] [--test PATH] [--iterations N]
                                 Texel-tune the hand-written eval terms
  chessgpu help                  this text

options:
  --fen "<FEN>"            position to search (default: the start position)
  --moves "<moves>"        UCI moves played from that position first
  --board quad|std         board representation (default: quad if compiled in)
  --model cpu|thread|warp  CPU reference, or a GPU execution model (default: thread if compiled in)
  --attacks hq|magic|kogge sliding-piece attacks in the GPU kernels (default: hq; the search has hq and
                           magic); perft's bench --all runs every compiled one
  --nnue-file PATH         NNUE weights for eval nnue instead of the compiled-in network (any command,
                           uci included: chessgpu uci --nnue-file PATH)
  --move-bits 16|32        perft's move encoding, CPU and GPU (default: 16; 32 adds the moving and
                           captured pieces); bench --all runs every compiled one
  --split N                plies expanded on the CPU before the GPU takes over
                           (perft default: automatic; search default: 2)
  --tasks-per-worker N     tasks per GPU worker for the automatic split (default: 8)
  --block-size N           GPU threads per block, a multiple of 32 (default: 128)
  --divide                 also print the count for each root move
  --file PATH              perft suite file (default: tests/perft/standard.epd)
  --max-nodes N            perft-suite: skip expected counts above N (default: 1000000000)
  --repeat N               bench: runs per position, best time kept (default: 1)
  --all                    bench: every GPU board/model combination compiled into this build

search options:
  --search cpu|gpu-independent|ybwc-host|ybwc-device|abdada-cpu|abdada-device
                                 which search runs (default: ybwc-device; --model cpu means cpu)
  --leaf-depth N                 ybwc: deepest task (default 2)
  --explicit-plies N             ybwc: plies kept as explicit nodes at most (default 8)
  --cut-waves / --no-cut-waves   ybwc: young brothers of CUT nodes in growing waves (default: on)
  --node-budget N                ybwc: split tasks that search more than N nodes (default 2048; 0: never)
  --lanes N                      ybwc-device and abdada-device: working threads per warp (default 1)
  --workers N                    abdada-device: workers, capped at what fits on the GPU (default 2048)
  --aspiration CP                aspiration windows from depth 4, first margin CP (default 0: full windows)
  --abdada-min-depth N           abdada: plies left from which moves are claimed and deferred (default 2)
  --abdada-defer-first 0|1       abdada: defer a node's first move too while another worker searches it (default 0)
  --threads N                    abdada-cpu and selfplay: CPU threads (default 4)
  --eval pesto|material|terms|nnue  evaluation (default: the engine's, {EVAL})
  --ordering staged|list         move ordering implementation, CPU and thread model (default: staged)
  --stack local|shared           thread-model stack placement (default: local)
{FEATURE_LIST}                                 pruning feature set (default: the engine's, {FEATURES})
  --tt-layout packed64|wide128   transposition table entry layout (default: wide128)
  --hash MB                      transposition table size (default: 2048)
  --null-min-depth N             null move: fewest plies left to try one (default: 2)
  --null-reduction R             null move: depth reduction R (default: 3) ...
  --null-divisor D               ... plus depth / D (default: 6; 0: none)
  --lmr-min-depth N              late move reductions: fewest plies left to reduce (default: 3)
  --lmr-moves N                  ... moves searched first without reductions (default: 3)
  --lmr-base B --lmr-divisor D   ... R = B / 100 + ln(depth) ln(move) / (D / 100) (default: 75, 225)
  --fut-depth N                  futility pruning: most plies left to prune at (default: 3)
  --fut-margin M                 ... margin per ply left, centipawns (default: 150)
  --delta-margin M               quiescence delta pruning margin, centipawns (default: 200; 0: off)
  --see-prune 0|1                quiescence SEE pruning off or on (default: 1)
  --depth N                      deepest iteration (default: 63, or until another limit)
  --movetime MS                  time limit
  --nodes N                      node limit, checked between iterations
)";

struct Options {
    std::string command;
    std::vector<std::string> positional;
    std::string fen = kStartFen;
    std::string moves;
    BoardKind board = BoardKind::Quad;
    bool cpu = false;  // --model cpu
    GpuPerftOptions gpu;
    bool divide = false;
    std::string file = CHESSGPU_SOURCE_DIR "/tests/perft/standard.epd";
    u64 max_nodes = 1'000'000'000;
    int repeat = 1;
    bool all = false;
    // search
    SearchKind search = SearchKind::YbwcDevice;
    EvalKind eval = kEngineEval;
    OrderingKind ordering = OrderingKind::Staged;
    StackKind stack = StackKind::Local;
    FeatureKind features = kEngineFeatures;
    TTLayoutKind tt_layout = TTLayoutKind::Wide128;
    size_t hash_mb = kDefaultHashMb;
    PruneParams prune;
    int depth = 0;
    double movetime_ms = -1;
    u64 nodes = 0;
    YbwcOptions ybwc;
    int lanes = 1;
    bool file_given = false;
    int limit = 0;
    int iterations = 0;  // tune
    // selfplay
    int games = 1000;
    int threads = 4;
    unsigned workers = 2048;
    AbdadaParams abdada{};
    int aspiration = 0;
    int random_plies = 4;
    unsigned long long seed = 1;
    bool hash_given = false;
    std::string test_file;  // tune: positions judged only at the end
    // train-nnue
    int epochs = 10;
    int hidden = kNnueHidden;  // train-nnue: neurons per side (only kNnueHidden gets written)
    double lambda = 0.75;  // weight of the search score against the game's result
    double lr = 1e-3;
    double weight_decay = 0;  // train-nnue: AdamW decay of the weights (0 = off)
    int batch = 16384;
    std::string out_path;
    std::string header_path;
    std::string nnue_file;  // --nnue-file: weights loaded before anything runs
};

// "  --features a|b|c...", every feature set this build knows, wrapped.
std::string feature_choices() {
    std::string out;
    std::string line = "  --features ";
    size_t i = 0;
    const size_t count = std::size(kAllFeatureKinds);
    for (const FeatureKind f : kAllFeatureKinds) {
        const std::string name = std::string(name_of(f)) + (++i < count ? "|" : "");
        if (line.size() + name.size() > 110) {
            out += line + "\n";
            line = "             ";
        }
        line += name;
    }
    return out + line + "\n";
}

std::string usage() {
    std::string text = kUsage;
    const auto fill = [&](const std::string& key, const std::string& value) {
        const size_t at = text.find(key);
        if (at != std::string::npos) text.replace(at, key.size(), value);
    };
    fill("{EVAL}", name_of(kEngineEval));
    fill("{FEATURES}", name_of(kEngineFeatures));
    fill("{FEATURE_LIST}", feature_choices());
    return text;
}

[[noreturn]] void usage_error(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), usage().c_str());
    std::exit(2);
}

long long parse_integer(const std::string& text, const char* what) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used);
        if (used == text.size()) return value;
    } catch (const std::exception&) {
    }
    usage_error(std::string("invalid ") + what + ": " + text);
}

Options parse_args(int argc, char** argv) {
    Options o;
    // Defaults: the current winners, if this build compiles them (CHESSGPU_BOARDS / CHESSGPU_MODELS).
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.gpu.model = is_compiled(ExecModel::Thread) ? ExecModel::Thread : compiled_models().front();
    o.command = argc < 2 ? "uci" : argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage_error(arg + " needs a value");
            return argv[++i];
        };
        if (arg == "--fen") {
            o.fen = value();
        } else if (arg == "--moves") {
            o.moves = value();
        } else if (arg == "--board") {
            const std::string v = value();
            if (!parse_board_kind(v, o.board)) usage_error("unknown board: " + v);
        } else if (arg == "--attacks") {
            const std::string v = value();
            if (!parse_attacks_kind(v, o.gpu.attacks)) usage_error("unknown attacks: " + v);
        } else if (arg == "--nnue-file") {
            o.nnue_file = value();
        } else if (arg == "--move-bits") {
            const std::string v = value();
            if (!parse_moves_kind(v, o.gpu.moves)) usage_error("unknown move encoding: " + v);
        } else if (arg == "--model") {
            const std::string v = value();
            o.cpu = v == "cpu";
            if (o.cpu) o.search = SearchKind::Cpu;
            else if (!parse_exec_model(v, o.gpu.model)) usage_error("unknown model: " + v);
        } else if (arg == "--search") {
            const std::string v = value();
            if (!parse_search_kind(v, o.search)) usage_error("unknown search: " + v);
        } else if (arg == "--eval") {
            const std::string v = value();
            if (!parse_eval_kind(v, o.eval)) usage_error("unknown eval: " + v);
        } else if (arg == "--ordering") {
            const std::string v = value();
            if (!parse_ordering_kind(v, o.ordering)) usage_error("unknown ordering: " + v);
        } else if (arg == "--stack") {
            const std::string v = value();
            if (!parse_stack_kind(v, o.stack)) usage_error("unknown stack: " + v);
        } else if (arg == "--features") {
            const std::string v = value();
            if (!parse_feature_kind(v, o.features)) usage_error("unknown feature set: " + v);
        } else if (arg == "--tt-layout") {
            const std::string v = value();
            if (!parse_tt_layout_kind(v, o.tt_layout)) usage_error("unknown table layout: " + v);
        } else if (arg == "--hash") {
            const long long mb = parse_integer(value(), "hash size");
            if (mb < 1 || mb > (long long)kMaxHashMb) usage_error("--hash must be between 1 and 65536");
            o.hash_mb = size_t(mb);
            o.hash_given = true;
        } else if (arg == "--null-min-depth") {
            const long long d = parse_integer(value(), "null move minimum depth");
            if (d < 1 || d > kMaxNullMinDepth) usage_error("--null-min-depth must be between 1 and 20");
            o.prune.null_min_depth = u8(d);
        } else if (arg == "--null-reduction") {
            const long long r = parse_integer(value(), "null move reduction");
            if (r < 1 || r > kMaxNullBase) usage_error("--null-reduction must be between 1 and 10");
            o.prune.null_base = u8(r);
        } else if (arg == "--null-divisor") {
            const long long d = parse_integer(value(), "null move depth divisor");
            if (d < 0 || d > kMaxNullDivisor) usage_error("--null-divisor must be between 0 and 20");
            o.prune.null_divisor = u8(d);
        } else if (arg == "--lmr-min-depth") {
            const long long d = parse_integer(value(), "late move reduction minimum depth");
            if (d < 1 || d > kMaxLmrMinDepth) usage_error("--lmr-min-depth must be between 1 and 20");
            o.prune.lmr_min_depth = u8(d);
        } else if (arg == "--lmr-moves") {
            const long long m = parse_integer(value(), "moves before late move reductions");
            if (m < 0 || m > kMaxLmrMoves) usage_error("--lmr-moves must be between 0 and 63");
            o.prune.lmr_moves = u8(m);
        } else if (arg == "--lmr-base") {
            const long long b = parse_integer(value(), "late move reduction base");
            if (b < 0 || b > 255) usage_error("--lmr-base must be between 0 and 255");
            o.prune.lmr_base = u8(b);
        } else if (arg == "--lmr-divisor") {
            const long long d = parse_integer(value(), "late move reduction divisor");
            if (d < 1 || d > 255) usage_error("--lmr-divisor must be between 1 and 255");
            o.prune.lmr_divisor = u8(d);
        } else if (arg == "--fut-depth") {
            const long long d = parse_integer(value(), "futility pruning depth");
            if (d < 1 || d > kMaxFutDepth) usage_error("--fut-depth must be between 1 and 10");
            o.prune.fut_depth = u8(d);
        } else if (arg == "--fut-margin") {
            const long long m = parse_integer(value(), "futility margin");
            if (m < 0 || m > 255) usage_error("--fut-margin must be between 0 and 255");
            o.prune.fut_margin = u8(m);
        } else if (arg == "--delta-margin") {
            const long long m = parse_integer(value(), "delta pruning margin");
            if (m < 0 || m > kMaxDeltaMargin) usage_error("--delta-margin must be between 0 and 2000");
            o.prune.delta_margin = u16(m);
        } else if (arg == "--see-prune") {
            const long long s = parse_integer(value(), "SEE pruning");
            if (s < 0 || s > 1) usage_error("--see-prune must be 0 or 1");
            o.prune.see_prune = u8(s);
        } else if (arg == "--leaf-depth") {
            o.ybwc.leaf_depth = int(parse_integer(value(), "leaf depth"));
        } else if (arg == "--lanes") {
            o.lanes = int(parse_integer(value(), "lanes"));
        } else if (arg == "--abdada-min-depth") {
            const long long d = parse_integer(value(), "ABDADA minimum depth");
            if (d < 1 || d > 20) usage_error("--abdada-min-depth must be between 1 and 20");
            o.abdada.min_depth = u8(d);
        } else if (arg == "--abdada-defer-first") {
            const long long d = parse_integer(value(), "ABDADA defer-first");
            if (d < 0 || d > 1) usage_error("--abdada-defer-first must be 0 or 1");
            o.abdada.defer_first = u8(d);
        } else if (arg == "--aspiration") {
            const long long m = parse_integer(value(), "aspiration");
            if (m < 0 || m > 1000) usage_error("--aspiration must be between 0 and 1000");
            o.aspiration = int(m);
        } else if (arg == "--workers") {
            const long long w = parse_integer(value(), "workers");
            if (w < 1 || w > 1048576) usage_error("--workers must be between 1 and 1048576");
            o.workers = unsigned(w);
        } else if (arg == "--node-budget") {
            o.ybwc.node_budget = u32(parse_integer(value(), "node budget"));
        } else if (arg == "--cut-waves") {
            o.ybwc.cut_waves = true;
        } else if (arg == "--no-cut-waves") {
            o.ybwc.cut_waves = false;
        } else if (arg == "--explicit-plies") {
            o.ybwc.max_explicit_ply = int(parse_integer(value(), "explicit plies"));
        } else if (arg == "--test") {
            o.test_file = value();
        } else if (arg == "--hidden") {
            o.hidden = int(parse_integer(value(), "hidden"));
            if (o.hidden < 1 || o.hidden > kNnueMaxHidden) usage_error("--hidden must be between 1 and 512");
        } else if (arg == "--epochs") {
            o.epochs = int(parse_integer(value(), "epochs"));
            if (o.epochs < 1 || o.epochs > 1000) usage_error("--epochs must be between 1 and 1000");
        } else if (arg == "--lambda") {
            o.lambda = std::atof(value().c_str());
            if (o.lambda < 0 || o.lambda > 1) usage_error("--lambda must be between 0 and 1");
        } else if (arg == "--weight-decay") {
            o.weight_decay = std::atof(value().c_str());
            if (o.weight_decay < 0 || o.weight_decay > 100) usage_error("--weight-decay must be between 0 and 100");
        } else if (arg == "--lr") {
            o.lr = std::atof(value().c_str());
            if (o.lr <= 0 || o.lr > 1) usage_error("--lr must be in (0, 1]");
        } else if (arg == "--batch") {
            o.batch = int(parse_integer(value(), "batch"));
            if (o.batch < 1) usage_error("--batch must be positive");
        } else if (arg == "--out") {
            o.out_path = value();
        } else if (arg == "--header") {
            o.header_path = value();
        } else if (arg == "--games") {
            o.games = int(parse_integer(value(), "games"));
            if (o.games < 1) usage_error("--games must be at least 1");
        } else if (arg == "--threads") {
            o.threads = int(parse_integer(value(), "threads"));
            if (o.threads < 1 || o.threads > 64) usage_error("--threads must be between 1 and 64");
        } else if (arg == "--random-plies") {
            o.random_plies = int(parse_integer(value(), "random plies"));
            if (o.random_plies < 0 || o.random_plies > 20) usage_error("--random-plies must be between 0 and 20");
        } else if (arg == "--seed") {
            o.seed = (unsigned long long)parse_integer(value(), "seed");
        } else if (arg == "--iterations") {
            o.iterations = int(parse_integer(value(), "iterations"));
            if (o.iterations < 1) usage_error("--iterations must be at least 1");
        } else if (arg == "--depth") {
            o.depth = int(parse_integer(value(), "depth"));
            if (o.depth < 1 || o.depth >= kMaxPly) usage_error("--depth must be between 1 and 63");
        } else if (arg == "--movetime") {
            o.movetime_ms = double(parse_integer(value(), "move time"));
        } else if (arg == "--nodes") {
            o.nodes = u64(parse_integer(value(), "node limit"));
        } else if (arg == "--split") {
            o.gpu.split_depth = int(parse_integer(value(), "split depth"));
        } else if (arg == "--tasks-per-worker") {
            o.gpu.tasks_per_worker = int(parse_integer(value(), "tasks per worker"));
        } else if (arg == "--block-size") {
            o.gpu.block_size = int(parse_integer(value(), "block size"));
        } else if (arg == "--divide") {
            o.divide = true;
        } else if (arg == "--file") {
            o.file = value();
            o.file_given = true;
        } else if (arg == "--limit") {
            o.limit = int(parse_integer(value(), "limit"));
        } else if (arg == "--max-nodes") {
            o.max_nodes = u64(parse_integer(value(), "node limit"));
        } else if (arg == "--repeat") {
            o.repeat = int(parse_integer(value(), "repeat count"));
            if (o.repeat < 1) usage_error("--repeat must be at least 1");
        } else if (arg == "--all") {
            o.all = true;
        } else if (arg.rfind("--", 0) == 0) {
            usage_error("unknown option: " + arg);
        } else {
            o.positional.push_back(arg);
        }
    }
    return o;
}

Position parse_position_or_exit(const std::string& fen) {
    Position p;
    std::string error;
    if (!parse_fen(fen, p, error)) usage_error("invalid FEN \"" + fen + "\": " + error);
    return p;
}

std::string with_commas(u64 n) {
    std::string digits = std::to_string(n);
    for (int i = int(digits.size()) - 3; i > 0; i -= 3) digits.insert(size_t(i), ",");
    return digits;
}

// Nodes per second, scaled to G/M/k.
std::string rate(u64 nodes, double ms) {
    if (ms <= 0) return "-";
    const double per_second = double(nodes) / (ms / 1000.0);
    char text[32];
    if (per_second >= 1e9) std::snprintf(text, sizeof text, "%.2f Gnodes/s", per_second / 1e9);
    else if (per_second >= 1e6) std::snprintf(text, sizeof text, "%.2f Mnodes/s", per_second / 1e6);
    else std::snprintf(text, sizeof text, "%.0f nodes/s", per_second);
    return text;
}

struct PerftRun {
    u64 nodes = 0;
    std::vector<std::pair<Move, u64>> divide;
    GpuPerftStats stats;  // GPU runs only
    double ms = 0;        // wall time of the whole run
    double work_ms = 0;   // CPU: same as ms; GPU: kernel time
};

PerftRun run_perft(const Options& o, BoardKind board, ExecModel model, AttacksKind attacks, MovesKind moves,
                   const DeviceContext* ctx, const Position& p, int depth, bool divide) {
    PerftRun r;
    if (o.cpu) {
        const auto with_attacks = [&](const auto& a) {
            const Stopwatch timer;
            const auto run_with = [&]<class Moves>(auto b) {
                if (divide && depth >= 1) {
                    r.divide = perft_divide<Moves>(b, a, depth);
                    for (const auto& entry : r.divide) r.nodes += entry.second;
                } else {
                    r.nodes = perft_recursive<Moves>(b, a, depth);
                }
            };
            const auto run = [&](auto b) {
                if (moves == MovesKind::Bits32) run_with.template operator()<Moves32>(b);
                else run_with.template operator()<Moves16>(b);
            };
            if (board == BoardKind::Std) run(to_board<StdBoard>(p));
            else run(to_board<QuadBoard>(p));
            r.ms = r.work_ms = timer.elapsed_ms();
        };
        if (attacks == AttacksKind::Magic) with_attacks(MagicAttacks(host_attack_tables(), host_magic_tables()));
        else if (attacks == AttacksKind::Kogge) with_attacks(KoggeStoneAttacks(host_attack_tables()));
        else with_attacks(HqAttacks(host_attack_tables()));
    } else {
        GpuPerftOptions g = o.gpu;
        g.board = board;
        g.model = model;
        g.attacks = attacks;
        g.moves = moves;
        g.divide = divide;
        GpuPerftResult gr = gpu_perft(*ctx, p, depth, g);
        r.nodes = gr.nodes;
        r.divide = std::move(gr.divide);
        r.stats = gr.stats;
        r.ms = gr.stats.total_ms;
        r.work_ms = gr.stats.kernel_ms;
    }
    return r;
}

std::string describe_run(const Options& o, BoardKind board, ExecModel model, const PerftRun& r) {
    char text[512];
    const std::string attacks = std::string(name_of(o.gpu.attacks)) +
                                (o.gpu.moves == MovesKind::Bits32 ? " | 32-bit moves" : "");
    if (o.cpu) {
        std::snprintf(text, sizeof text, "%s | cpu | %s | %.1f ms | %s", name_of(board), attacks.c_str(), r.ms,
                      rate(r.nodes, r.ms).c_str());
    } else {
        const GpuPerftStats& s = r.stats;
        std::snprintf(text, sizeof text,
                      "%s | %s | %s | split %d (%zu tasks) | %d blocks x %d | kernel %.2f ms, %s | total %.1f ms "
                      "(split %.1f, upload %.1f, download %.1f)",
                      name_of(board), name_of(model), attacks.c_str(), s.split_depth, s.tasks, s.blocks,
                      s.block_size, s.kernel_ms, rate(r.nodes, s.kernel_ms).c_str(), s.total_ms, s.split_ms,
                      s.upload_ms, s.download_ms);
    }
    return text;
}

int cmd_uci() {
    UciEngine engine(std::cout);
    for (std::string line; std::getline(std::cin, line);)
        if (!engine.handle(line)) return 0;
    engine.handle("quit");  // end of input: like "quit"
    return 0;
}

// Positions of a "FEN ; comment" file ('#' starts a comment line).
std::vector<std::pair<std::string, std::string>> read_position_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage_error("can't open " + path);
    std::vector<std::pair<std::string, std::string>> out;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const size_t semi = line.find(';');
        std::string fen = line.substr(0, semi);
        std::string comment = semi == std::string::npos ? "" : line.substr(semi + 1);
        while (!fen.empty() && fen.back() == ' ') fen.pop_back();
        while (!comment.empty() && comment.front() == ' ') comment.erase(comment.begin());
        out.emplace_back(fen, comment);
    }
    return out;
}

EngineOptions engine_options(const Options& o) {
    EngineOptions eo;
    eo.search = o.search;
    eo.gpu.board = o.board;
    eo.gpu.model = o.gpu.model;
    eo.gpu.eval = o.eval;
    eo.gpu.ordering = o.ordering;
    eo.gpu.stack = o.stack;
    eo.gpu.features = o.features;
    eo.gpu.tt_layout = o.tt_layout;
    eo.gpu.prune = o.prune;
    eo.gpu.attacks = o.gpu.attacks;
    eo.threads = o.threads;
    eo.workers = o.workers;
    eo.abdada = o.abdada;
    eo.aspiration_cp = o.aspiration;
    eo.hash_mb = o.hash_mb;
    eo.gpu.block_size = o.gpu.block_size;
    eo.cpu_ordering = o.ordering;
    eo.split_depth = o.gpu.split_depth >= 0 ? o.gpu.split_depth : 2;
    eo.ybwc = o.ybwc;
    eo.device_limits.lanes = o.lanes;
    return eo;
}

// Texel tuning of the evaluation terms (host/tune.h): fits the weights of
// core/eval_terms.h to the results of the games in the tuning set and prints
// them as term_weight()'s switch body.
int cmd_tune(const Options& o) {
    const std::string path = o.file_given ? o.file : CHESSGPU_SOURCE_DIR "/build/tuning/positions.txt";
    const Stopwatch timer;
    const TuningSet set = load_tuning_set(path);
    std::printf("tune: %zu positions read, %zu not quiet; %zu for training, %zu for validation (%.1f s)\n",
                set.lines, set.rejected, set.training.size(), set.validation.size(), timer.elapsed_ms() / 1000);
    if (set.training.empty()) return 1;
    const TuningPositions& check = set.validation.empty() ? set.training : set.validation;
    const TuneWeights start = current_term_weights();
    const TuneWeights zero{};
    const double k = fit_scale(set.training, start);
    std::printf("scale k = %.4f | validation loss: PeSTO alone %.6f, current terms %.6f\n", k,
                tuning_loss(check, zero, k), tuning_loss(check, start, k));
    // Each strength of the pull toward the current weights, judged on the
    // validation games; the best one's weights are printed.
    TuneOptions options;
    if (o.iterations > 0) options.iterations = o.iterations;
    options.anchor = start;
    TuneWeights best = start;
    double best_loss = tuning_loss(check, start, k);
    for (const double l2 : {0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5}) {
        options.l2 = l2;
        const TuneWeights tuned = tune_weights(set.training, k, start, options);
        const double loss = tuning_loss(check, tuned, k);
        std::printf("l2 %.0e: training loss %.6f, validation loss %.6f\n", l2, tuning_loss(set.training, tuned, k),
                    loss);
        if (loss < best_loss) {
            best_loss = loss;
            best = tuned;
        }
    }
    std::printf("best validation loss %.6f (%.1f s); weights:\n%s", best_loss, timer.elapsed_ms() / 1000,
                format_term_weights(best).c_str());
    if (!o.test_file.empty()) {
        // Positions from elsewhere (e.g. match games when tuning on self-play),
        // all of them a test: their own scale, then each set of weights.
        const TuningSet t = load_tuning_set(o.test_file);
        TuningPositions all = t.training;
        all.insert(all.end(), t.validation.begin(), t.validation.end());
        const double tk = fit_scale(all, start);
        std::printf("test set %s: %zu quiet positions, scale %.4f | loss: PeSTO alone %.6f, current terms %.6f, "
                    "tuned %.6f\n",
                    o.test_file.c_str(), all.size(), tk, tuning_loss(all, zero, tk), tuning_loss(all, start, tk),
                    tuning_loss(all, best, tk));
    }
    return 0;
}

// Self-play games with the CPU search for tuning data (host/selfplay.h),
// appended to build/tuning/selfplay.txt (or --file), numbered on from the
// games already there.
// Labels positions for NNUE training: reads "<FEN> | <result> | <game>"
// lines (a tuning set or older self-play), searches each position to a fixed
// depth on the CPU (several threads, a table per thread cleared for every
// position, so a label doesn't depend on the order) and writes the lines
// with white's score as a fourth field, in the input's order.
int cmd_label(const Options& o) {
    if (!o.file_given || o.out_path.empty()) usage_error("label needs --file IN and --out OUT");
    std::ifstream in(o.file);
    if (!in) usage_error("can't read " + o.file);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) lines.push_back(line);
    const int depth = o.depth > 0 ? o.depth : 6;
    CpuSearchOptions search{o.board, o.eval, o.ordering};
    search.features = o.features;
    search.tt_layout = o.tt_layout;
    search.prune = o.prune;
    std::printf("label: %zu positions of %s, depth %d, %d threads, %s -> %s\n", lines.size(), o.file.c_str(), depth,
                o.threads, describe(search).c_str(), o.out_path.c_str());
    std::fflush(stdout);
    std::vector<std::string> scores(lines.size());
    std::atomic<size_t> next{0}, done{0};
    std::atomic<bool> failed{false};
    std::string failure;
    std::mutex mutex;
    const Stopwatch timer;
    const auto worker = [&] {
        try {
            CpuBackend backend(search, o.hash_given ? o.hash_mb : 16);
            for (size_t i = next++; i < lines.size() && !failed; i = next++) {
                const std::string fen = lines[i].substr(0, lines[i].find('|'));
                Position p;
                std::string error;
                if (!parse_fen(fen, p, error)) continue;  // left unlabeled
                SearchRequest request;
                request.root = p;
                request.max_depth = depth;
                backend.clear_hash();
                const SearchOutcome s = iterative_deepening(backend, request, {});
                const int white = state::side(p.state) == WHITE ? s.score : -s.score;
                scores[i] = std::to_string(white);
                const size_t n = ++done;
                if (n % 100000 == 0) {
                    const std::lock_guard<std::mutex> lock(mutex);
                    std::printf("%zu positions, %.0f s\n", n, timer.elapsed_ms() / 1000);
                    std::fflush(stdout);
                }
            }
        } catch (const std::exception& e) {
            const std::lock_guard<std::mutex> lock(mutex);
            failure = e.what();
            failed = true;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < o.threads; ++t) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
    if (failed) usage_error("label failed: " + failure);
    std::ofstream out(o.out_path);
    if (!out) usage_error("can't write " + o.out_path);
    size_t written = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (scores[i].empty()) continue;
        out << lines[i] << " | " << scores[i] << '\n';
        ++written;
    }
    std::printf("%zu positions labeled in %.0f s\n", written, timer.elapsed_ms() / 1000);
    return 0;
}

// NNUE training (host/nnue_train.h): fits the network to scored self-play
// positions, prints the losses per epoch, writes the quantized weights (and,
// with --header, the C++ file the engine's default net comes from:
// src/host/nnue_default.cpp).
int cmd_train_nnue(const Options& o) {
    const std::string path = o.file_given ? o.file : CHESSGPU_SOURCE_DIR "/build/nnue/selfplay.txt";
    const Stopwatch timer;
    std::string error;
    const NnueDataSet data = load_nnue_data(path, float(o.lambda), 3000, error);
    if (!error.empty()) usage_error(error);
    std::printf("train-nnue: %s: %zu positions for training, %zu for validation, %lld skipped (%.1f s); "
                "%dx%d network, lambda %.2f\n",
                path.c_str(), data.train.size(), data.validation.size(), data.skipped, timer.elapsed_ms() / 1000,
                kNnueInputs, o.hidden, o.lambda);
    if (data.train.empty()) return 1;
    NnueTrainOptions t;
    t.epochs = o.epochs;
    t.batch = o.batch;
    t.lr = float(o.lr);
    t.threads = o.threads;
    t.seed = o.seed;
    t.weight_decay = float(o.weight_decay);
    NnueFloat net = NnueFloat::random(o.seed, o.hidden);
    std::printf("start: validation loss %.6f\n", nnue_loss(net, data.validation, t.threads));
    std::fflush(stdout);
    train_nnue(net, data, t, [](const NnueEpochStats& s) {
        std::printf("epoch %d: training loss %.6f, validation loss %.6f (%.1f s)\n", s.epoch, s.train_loss,
                    s.validation_loss, s.seconds);
        std::fflush(stdout);
    });
    if (net.hidden != kNnueHidden) {
        std::printf("not written: the search's network is %dx%d (%.1f s in all)\n", kNnueInputs, kNnueHidden,
                    timer.elapsed_ms() / 1000);
        return 0;
    }
    const NnueWeights w = quantize_nnue(net);
    const std::string out = o.out_path.empty() ? CHESSGPU_SOURCE_DIR "/build/nnue/net.nnue" : o.out_path;
    if (!save_nnue(out, w, error)) usage_error(error);
    std::printf("weights: %s (%.1f s in all)\n", out.c_str(), timer.elapsed_ms() / 1000);
    if (!o.header_path.empty()) {
        char comment[256];
        std::snprintf(comment, sizeof comment, "%zu positions of %s, %d epochs, lambda %.2f, lr %g, batch %d",
                      data.train.size(), path.c_str(), o.epochs, o.lambda, o.lr, o.batch);
        if (!write_nnue_header(o.header_path, w, comment, error)) usage_error(error);
        std::printf("C++ weights: %s\n", o.header_path.c_str());
    }
    return 0;
}

int cmd_selfplay(const Options& o) {
    SelfPlayOptions s;
    s.games = o.games;
    s.depth = o.depth > 0 ? o.depth : 5;
    s.threads = o.threads;
    s.random_plies = o.random_plies;
    s.seed = o.seed;
    s.search = CpuSearchOptions{o.board, o.eval, o.ordering};
    s.search.features = o.features;
    s.search.tt_layout = o.tt_layout;
    s.search.prune = o.prune;
    s.hash_mb = o.hash_given ? o.hash_mb : 16;
    s.out_path = o.file_given ? o.file : CHESSGPU_SOURCE_DIR "/build/tuning/selfplay.txt";
    {
        std::ifstream openings(CHESSGPU_SOURCE_DIR "/tests/search/openings.epd");
        for (std::string line; std::getline(openings, line);)
            if (!line.empty() && line[0] != '#') s.openings.push_back(line.substr(0, line.find(" ;")));
        // Continue the numbering of an existing file.
        std::ifstream existing(s.out_path);
        for (std::string line; std::getline(existing, line);) {
            // The game number is the third field.
            const size_t bar = line.find('|');
            const size_t bar2 = bar == std::string::npos ? bar : line.find('|', bar + 1);
            if (bar2 != std::string::npos) s.first_game = std::max(s.first_game, std::atoi(line.c_str() + bar2 + 1) + 1);
        }
    }
    std::printf("selfplay: %d games, depth %d, %d threads, %s, %zu openings + %d random plies -> %s (games from %d)\n",
                s.games, s.depth, s.threads, describe(s.search).c_str(), s.openings.size(), s.random_plies,
                s.out_path.c_str(), s.first_game);
    const Stopwatch timer;
    const SelfPlayStats stats = run_selfplay(s, [&](const SelfPlayStats& st) {
        if (st.games % 100 == 0 || st.games == s.games)
            std::printf("%d games (+%d -%d =%d), %llu positions, %.0f s\n", st.games, st.white_wins, st.black_wins,
                        st.draws, st.positions, timer.elapsed_ms() / 1000);
        std::fflush(stdout);
    });
    std::printf("done: %d games, %llu positions in %.0f s\n", stats.games, stats.positions, timer.elapsed_ms() / 1000);
    return 0;
}

int cmd_bench_search(const Options& o) {
    const std::string file = o.file_given ? o.file : CHESSGPU_SOURCE_DIR "/tests/search/suite.epd";
    const auto positions = read_position_file(file);
    const int depth = o.depth > 0 ? o.depth : 8;
    const EngineOptions eo = engine_options(o);
    std::unique_ptr<DeviceContext> ctx;
    if (needs_gpu(eo.search)) ctx = std::make_unique<DeviceContext>();
    const std::unique_ptr<SearchBackend> backend = make_backend(eo, ctx.get());
    std::printf("bench-search: %s | depth %d | %s\n", backend->describe().c_str(), depth, file.c_str());
    double total_ms = 0;
    u64 total_nodes = 0;
    size_t count = 0;
    for (const auto& [fen, comment] : positions) {
        if (o.limit > 0 && count >= size_t(o.limit)) break;
        SearchRequest request;
        std::string error;
        if (!build_game(parse_position_or_exit(fen), "", request.root, request.history, error)) usage_error(error);
        request.max_depth = depth;
        request.aspiration_cp = o.aspiration;
        backend->clear_stop();
        backend->clear_hash();  // every position from an empty table, like a new game
        const SearchOutcome out = iterative_deepening(*backend, request, {});
        ++count;
        total_ms += out.ms;
        total_nodes += out.nodes;
        std::printf("%3zu  %9.1f ms  %15s nodes  depth %2d  %-12s %-6s  %s\n", count, out.ms,
                    with_commas(out.nodes).c_str(), out.depth, uci_score(out.score).c_str(),
                    out.best.is_none() ? "(none)" : move_to_uci(out.best).c_str(), comment.c_str());
        std::fflush(stdout);
        if (out.depth != depth) std::printf("     WARNING: reached depth %d only\n", out.depth);
    }
    std::printf("TOTAL %zu positions | %.1f ms | %s nodes | %s\n", count, total_ms, with_commas(total_nodes).c_str(),
                rate(total_nodes, total_ms).c_str());
    return 0;
}

int cmd_search(const Options& o) {
    if (!o.positional.empty()) usage_error("search takes no positional arguments");
    Position start = parse_position_or_exit(o.fen);
    SearchRequest request;
    std::string error;
    if (!build_game(start, o.moves, request.root, request.history, error)) usage_error(error);
    const EngineOptions eo = engine_options(o);
    std::unique_ptr<DeviceContext> ctx;
    if (needs_gpu(eo.search)) ctx = std::make_unique<DeviceContext>();
    const std::unique_ptr<SearchBackend> backend = make_backend(eo, ctx.get());
    if (o.depth > 0) request.max_depth = o.depth;
    request.max_nodes = o.nodes;
    request.aspiration_cp = o.aspiration;
    if (o.movetime_ms >= 0) {
        request.soft = request.hard =
            Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(o.movetime_ms));
    }
    if (o.depth == 0 && o.movetime_ms < 0 && o.nodes == 0) usage_error("search needs --depth, --movetime or --nodes");
    std::printf("search %s\nposition %s\n", backend->describe().c_str(), format_fen(request.root).c_str());
    const SearchOutcome out = iterative_deepening(*backend, request, [&](const IterationReport& r) {
        std::string pv;
        for (const Move m : r.pv) pv += " " + move_to_uci(m);
        std::printf("info depth %d score %s nodes %llu time %.0f nps %s pv%s\n", r.depth, uci_score(r.score).c_str(),
                    r.total_nodes, r.ms, rate(r.total_nodes, r.ms).c_str(), pv.c_str());
        const std::string stats = backend->last_stats();
        if (!stats.empty()) std::printf("     depth %d: %llu nodes | %s\n", r.depth, r.nodes, stats.c_str());
        std::fflush(stdout);
    });
    std::printf("bestmove %s\n", out.best.is_none() ? "(none)" : move_to_uci(out.best).c_str());
    std::printf("depth %d | %s nodes | %.1f ms | %s\n", out.depth, with_commas(out.nodes).c_str(), out.ms,
                rate(out.nodes, out.ms).c_str());
    return 0;
}

int cmd_info() {
    const DeviceContext ctx;
    std::printf("%s", describe(ctx.info()).c_str());
    return 0;
}

int cmd_perft(const Options& o) {
    if (o.positional.size() != 1) usage_error("perft needs exactly one depth");
    const int depth = int(parse_integer(o.positional[0], "depth"));
    if (depth < 0) usage_error("depth must be >= 0");
    Position p = parse_position_or_exit(o.fen);
    std::string error;
    if (!apply_uci_moves(p, o.moves, error)) usage_error(error);
    std::unique_ptr<DeviceContext> ctx;
    if (!o.cpu) ctx = std::make_unique<DeviceContext>();
    const PerftRun r = run_perft(o, o.board, o.gpu.model, o.gpu.attacks, o.gpu.moves, ctx.get(), p, depth, o.divide);
    for (const auto& [move, nodes] : r.divide) std::printf("%s: %llu\n", move_to_uci(move).c_str(), nodes);
    if (o.divide) std::printf("\n");
    std::printf("Nodes searched: %llu\n", r.nodes);
    std::printf("perft %d | %s\n", depth, describe_run(o, o.board, o.gpu.model, r).c_str());
    return 0;
}

int cmd_perft_suite(const Options& o) {
    std::vector<PerftCase> cases;
    std::string error;
    if (!load_perft_suite(o.file, cases, error)) usage_error(error);
    std::unique_ptr<DeviceContext> ctx;
    if (!o.cpu) ctx = std::make_unique<DeviceContext>();
    int passed = 0;
    int failed = 0;
    u64 total_nodes = 0;
    double total_work_ms = 0;
    for (const PerftCase& pc : cases) {
        Position p;
        if (!parse_fen(pc.fen, p, error)) {
            std::printf("FAIL line %d: invalid FEN (%s)\n", pc.line, error.c_str());
            ++failed;
            continue;
        }
        for (const auto& [depth, expected] : pc.expected) {
            if (expected > o.max_nodes) break;
            const PerftRun r =
                run_perft(o, o.board, o.gpu.model, o.gpu.attacks, o.gpu.moves, ctx.get(), p, depth, false);
            const bool ok = r.nodes == expected;
            ok ? ++passed : ++failed;
            total_nodes += r.nodes;
            total_work_ms += r.work_ms;
            std::printf("%s line %d D%d: %s nodes", ok ? "ok  " : "FAIL", pc.line, depth, with_commas(r.nodes).c_str());
            if (!ok) std::printf(" (expected %s)", with_commas(expected).c_str());
            std::printf(" | %.1f ms\n", r.ms);
            std::fflush(stdout);
        }
    }
    std::printf("\n%d passed, %d failed | %s nodes, %s (%s time)\n", passed, failed, with_commas(total_nodes).c_str(),
                rate(total_nodes, total_work_ms).c_str(), o.cpu ? "wall" : "kernel");
    return failed ? 1 : 0;
}

struct BenchPosition {
    const char* name;
    const char* fen;
    int gpu_depth;
    u64 gpu_nodes;
    int cpu_depth;
    u64 cpu_nodes;
};

// Fixed workload; every count is in tests/perft/standard.epd (Stockfish-verified).
constexpr BenchPosition kBench[] = {
    {"start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 7, 3195901860ull, 5, 4865609ull},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 6, 8031647685ull, 4,
     4085603ull},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 8, 3009794393ull, 6, 11030083ull},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033ull, 5, 15833292ull},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 6, 3048196529ull, 4, 2103487ull},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 6, 6923051137ull, 4,
     3894594ull},
};

int cmd_bench(const Options& o) {
    std::unique_ptr<DeviceContext> ctx;
    if (!o.cpu) ctx = std::make_unique<DeviceContext>();
    struct Combo {
        BoardKind board;
        ExecModel model;
        AttacksKind attacks;
        MovesKind moves;
    };
    std::vector<Combo> combos;
    if (o.all && !o.cpu) {
        for (const MovesKind mv : compiled_moves())
            for (const AttacksKind a : compiled_attacks())
                for (const BoardKind b : compiled_boards())
                    for (const ExecModel m : compiled_models()) combos.push_back({b, m, a, mv});
    } else {
        combos.push_back({o.board, o.gpu.model, o.gpu.attacks, o.gpu.moves});
    }

    bool all_correct = true;
    std::printf("%-6s %-7s %-6s %-5s %-9s %16s %12s %16s %12s\n", "board", "model", "attack", "moves", "position",
                "nodes", o.cpu ? "ms" : "kernel ms", "rate", "total ms");
    for (const auto& [board, model, attacks, moves] : combos) {
        u64 total_nodes = 0;
        double total_work = 0;
        double total_wall = 0;
        for (const BenchPosition& bp : kBench) {
            const Position p = parse_position_or_exit(bp.fen);
            const int depth = o.cpu ? bp.cpu_depth : bp.gpu_depth;
            const u64 expected = o.cpu ? bp.cpu_nodes : bp.gpu_nodes;
            PerftRun best;
            for (int i = 0; i < o.repeat; ++i) {
                PerftRun r = run_perft(o, board, model, attacks, moves, ctx.get(), p, depth, false);
                if (r.nodes != expected) {
                    std::printf("WRONG COUNT for %s depth %d: %llu, expected %llu\n", bp.name, depth, r.nodes, expected);
                    all_correct = false;
                }
                if (i == 0 || r.work_ms < best.work_ms) best = std::move(r);
            }
            total_nodes += best.nodes;
            total_work += best.work_ms;
            total_wall += best.ms;
            std::printf("%-6s %-7s %-6s %-5s %-9s %16s %12.2f %16s %12.1f\n", name_of(board),
                        o.cpu ? "cpu" : name_of(model), name_of(attacks), name_of(moves), bp.name,
                        with_commas(best.nodes).c_str(), best.work_ms, rate(best.nodes, best.work_ms).c_str(),
                        best.ms);
            std::fflush(stdout);
        }
        std::printf("%-6s %-7s %-6s %-5s %-9s %16s %12.2f %16s %12.1f\n\n", name_of(board),
                    o.cpu ? "cpu" : name_of(model), name_of(attacks), name_of(moves), "TOTAL",
                    with_commas(total_nodes).c_str(), total_work, rate(total_nodes, total_work).c_str(), total_wall);
    }
    return all_correct ? 0 : 1;
}

// Every compiled GPU search variant for the given board and eval.
std::vector<GpuSearchOptions> all_search_variants(int block_size) {
    std::vector<GpuSearchOptions> v;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material, EvalKind::Terms, EvalKind::Nnue}) {
            if (!is_compiled(eval)) continue;
            for (const ExecModel model : compiled_models()) {
                GpuSearchOptions o;
                o.board = board;
                o.eval = eval;
                o.model = model;
                o.block_size = block_size;
                if (model == ExecModel::Warp) {
                    v.push_back(o);
                    continue;
                }
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    if (!is_compiled(ordering)) continue;
                    o.ordering = ordering;
                    for (const StackKind stack : {StackKind::Local, StackKind::Shared}) {
                        if (ordering == OrderingKind::List && stack == StackKind::Shared) continue;
                        if (ordering == OrderingKind::Staged && !is_compiled(stack)) continue;
                        o.stack = stack;
                        v.push_back(o);
                    }
                }
            }
        }
    }
    return v;
}

// The positions `split` plies below each bench position, as full-window
// tasks of `depth` plies.
std::vector<TaskSpec> bench_tasks(int depth, int split) {
    const HqAttacks a(host_attack_tables());
    std::vector<TaskSpec> tasks;
    for (const BenchPosition& bp : kBench) {
        std::vector<QuadBoard> level = {to_board<QuadBoard>(parse_position_or_exit(bp.fen))};
        for (int ply = 0; ply < split; ++ply) {
            std::vector<QuadBoard> next;
            for (const QuadBoard& b : level) {
                MoveList list;
                generate_legal(b, a, list);
                for (int i = 0; i < list.size; ++i) {
                    QuadBoard c = b;
                    make_move(c, list.moves[i]);
                    next.push_back(c);
                }
            }
            level = std::move(next);
        }
        for (const QuadBoard& b : level) {
            TaskSpec t;
            t.board = b;
            t.depth = depth;
            t.ply = split;
            tasks.push_back(t);
        }
    }
    return tasks;
}

int cmd_bench_tasks(const Options& o) {
    const DeviceContext ctx;
    const int depth = o.depth > 0 ? o.depth : 3;
    const int split = o.gpu.split_depth >= 0 ? o.gpu.split_depth : 3;
    std::vector<TaskSpec> tasks = bench_tasks(depth, split);
    for (TaskSpec& t : tasks) t.node_budget = o.ybwc.node_budget;
    // One worker alone: the start position, two plies deeper.
    TaskSpec single;
    single.board = to_board<QuadBoard>(parse_position_or_exit(kStartFen));
    single.depth = depth + 2;
    std::vector<GpuSearchOptions> variants;
    if (o.all) {
        variants = all_search_variants(o.gpu.block_size);
    } else {
        GpuSearchOptions g;
        g.board = o.board;
        g.model = o.gpu.model;
        g.eval = o.eval;
        g.ordering = o.ordering;
        g.stack = o.stack;
        g.block_size = o.gpu.block_size;
        variants.push_back(g);
    }
    // With a table: every run starts from an empty one, so repeats measure the same work.
    std::unique_ptr<DeviceTT> table;
    if (uses_tt(o.features)) {
        table = std::make_unique<DeviceTT>(o.hash_mb);
        for (GpuSearchOptions& g : variants) {
            g.features = o.features;
            g.tt_layout = o.tt_layout;
            g.prune = o.prune;
        }
    }
    std::printf("%zu tasks: bench positions %d plies down, depth %d, full window, node budget %u; single task: start "
                "position depth %d\n",
                tasks.size(), split, depth, o.ybwc.node_budget, depth + 2);
    {
        // Task sizes, from one CPU-equivalent run on the GPU (every variant searches the same trees).
        GpuSearchOptions g;
        g.board = o.board;
        g.eval = o.eval;
        GpuSearcher sizer(ctx, g);
        const std::vector<SearchResult> res = sizer.run(tasks, {}, 0, {});
        std::vector<u64> sizes;
        for (const SearchResult& s : res) sizes.push_back(s.nodes);
        std::sort(sizes.begin(), sizes.end());
        const auto pct = [&](double q) { return sizes[size_t(q * double(sizes.size() - 1))]; };
        std::printf("task nodes: median %llu, 90%% %llu, 99%% %llu, 99.9%% %llu, max %llu\n", pct(0.5), pct(0.9),
                    pct(0.99), pct(0.999), sizes.back());
    }
    std::printf("%-44s %8s %14s %16s %12s %16s\n", "variant", "blocks", "nodes", "kernel ms", "rate", "single worker");
    for (GpuSearchOptions g : variants) {
        if (g.model == ExecModel::Warp && !o.all && o.gpu.block_size == 128) g.block_size = 64;
        GpuSearcher searcher(ctx, g);
        const auto fresh_table = [&] {
            if (!table) return;
            table->clear();
            searcher.set_tt(table->desc());
        };
        u64 nodes = 0;
        double best_ms = 0;
        for (int r = 0; r < o.repeat; ++r) {
            fresh_table();
            const std::vector<SearchResult> res = searcher.run(tasks, {}, 0, {});
            nodes = 0;
            for (const SearchResult& s : res) nodes += s.nodes;
            const double ms = searcher.last_batch_stats().kernel_ms;
            if (r == 0 || ms < best_ms) best_ms = ms;
        }
        const GpuBatchStats batch = searcher.last_batch_stats();
        double single_ms = 0;
        u64 single_nodes = 0;
        for (int r = 0; r < o.repeat; ++r) {
            fresh_table();
            const std::vector<SearchResult> res = searcher.run({single}, {}, 0, {});
            const double ms = searcher.last_batch_stats().kernel_ms;
            single_nodes = res[0].nodes;
            if (r == 0 || ms < single_ms) single_ms = ms;
        }
        std::printf("%-44s %3d x %-3d %14s %16.2f %12s %16s\n", describe(g).c_str(), batch.blocks_per_sm,
                    batch.block_size, with_commas(nodes).c_str(), best_ms, rate(nodes, best_ms).c_str(),
                    rate(single_nodes, single_ms).c_str());
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse_args(argc, argv);
    try {
        if (!o.nnue_file.empty()) {
            NnueWeights w;
            std::string error;
            if (!load_nnue(o.nnue_file, w, error)) usage_error(error);
            set_host_nnue_weights(w);
        }
        if (o.command == "uci") return cmd_uci();
        if (o.command == "search") return cmd_search(o);
        if (o.command == "bench-tasks") return cmd_bench_tasks(o);
        if (o.command == "bench-search") return cmd_bench_search(o);
        if (o.command == "info") return cmd_info();
        if (o.command == "perft") return cmd_perft(o);
        if (o.command == "perft-suite") return cmd_perft_suite(o);
        if (o.command == "bench") return cmd_bench(o);
        if (o.command == "tune") return cmd_tune(o);
        if (o.command == "selfplay") return cmd_selfplay(o);
        if (o.command == "train-nnue") return cmd_train_nnue(o);
        if (o.command == "label") return cmd_label(o);
        if (o.command == "help" || o.command == "--help" || o.command == "-h") {
            std::printf("%s", usage().c_str());
            return 0;
        }
        usage_error("unknown command: " + o.command);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
