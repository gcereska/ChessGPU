#pragma once

#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/device_context.h"
#include "host/position.h"
#include "host/variants.h"

namespace cb {

// Variants compiled into the GPU kernels (CMake options CHESSGPU_BOARDS and
// CHESSGPU_MODELS). The lists are in declaration order and never empty.
bool is_compiled(BoardKind k);
bool is_compiled(ExecModel m);
bool is_compiled(AttacksKind a);  // perft kernels only (CMake option CHESSGPU_ATTACKS)
bool is_compiled(MovesKind m);    // perft kernels (CMake option CHESSGPU_MOVES)
std::vector<BoardKind> compiled_boards();
std::vector<ExecModel> compiled_models();
std::vector<AttacksKind> compiled_attacks();
std::vector<MovesKind> compiled_moves();

struct GpuPerftOptions {
    BoardKind board = BoardKind::Quad;
    ExecModel model = ExecModel::Thread;
    AttacksKind attacks = AttacksKind::Hq;  // sliding-piece attacks in the kernels
    MovesKind moves = MovesKind::Bits16;    // move encoding in the kernels
    int split_depth = -1;       // plies expanded on the CPU before the GPU takes over; -1 = automatic
    int tasks_per_worker = 8;   // automatic split target: tasks per thread (thread model) or warp
    int block_size = 128;       // threads per block, a multiple of 32
    bool divide = false;        // also report the count for each root move
};

struct GpuPerftStats {
    int split_depth = 0;
    size_t tasks = 0;
    int blocks = 0;
    int block_size = 0;
    int blocks_per_sm = 0;
    double split_ms = 0;     // CPU: expanding the first plies into tasks
    double upload_ms = 0;    // copying tasks to the GPU
    double kernel_ms = 0;    // GPU time of the perft kernel (CUDA events)
    double download_ms = 0;  // copying results back and summing them
    double total_ms = 0;     // wall time of the whole call
};

struct GpuPerftResult {
    u64 nodes = 0;
    std::vector<std::pair<Move, u64>> divide;  // sorted by UCI notation; only with options.divide
    GpuPerftStats stats;
};

// Deepest subtree one GPU task searches. Deeper perfts split more plies on the CPU.
constexpr int kGpuMaxTaskDepth = 9;

// Perft on the GPU. The CPU expands the first plies breadth-first into tasks;
// a persistent kernel then works through them, one task per thread (thread
// model) or per warp (warp model). Throws std::invalid_argument for bad
// options, including a board or model that isn't compiled in.
GpuPerftResult gpu_perft(const DeviceContext& ctx, const Position& root, int depth, const GpuPerftOptions& options);

}  // namespace cb
