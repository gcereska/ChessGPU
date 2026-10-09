#pragma once

// YBWC run entirely on the GPU (docs/ARCHITECTURE.md, Parallel search
// on the GPU). The same node logic as the CPU-coordinated version
// (host/ybwc.h), but explicit nodes, their move lists and the queue of
// ready work live in device memory, and one persistent kernel runs the
// whole search: a worker that finishes a task updates the parent itself,
// starts young brothers, and carries finished results up the tree. No
// round trips to the CPU.
//
// Nondeterministic: results arrive in a different order each run, so
// windows, aborted subtrees and node counts vary. The score is still exactly
// the sequential search's, and so is the best move (without a transposition
// table; with one, workers also share results through it, see core/tt.h).

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "core/move_order.h"
#include "gpu/device_context.h"
#include "gpu/search_gpu.h"
#include "host/game_history.h"
#include "host/search_cpu.h"
#include "host/ybwc.h"

namespace cb {

// Pool sizes (allocated once, ~5.5 GB with the defaults) and the thread
// model's working lanes. A search that fills a pool comes back aborted with
// stats.overflow set; the engine then retries with coarser splitting.
struct YbwcDeviceLimits {
    int max_nodes = 1 << 25;        // explicit and task nodes per search (128 B each)
    int max_moves = 1 << 28;        // moves of explicit nodes per search
    int lanes = 1;                  // thread model: working threads per warp
};

struct YbwcDeviceStats {
    unsigned explicit_nodes = 0;
    unsigned tasks = 0;
    unsigned skipped = 0;      // queued tasks dropped because an ancestor had cut off
    unsigned over_budget = 0;  // tasks split for running out of node budget
    unsigned null_tries = 0;   // explicit nodes that searched a null move (tasks not counted)
    unsigned null_cuts = 0;    // ... and ended with its result
    unsigned lmr_reductions = 0;  // moves of explicit nodes searched reduced first (tasks not counted)
    unsigned lmr_researches = 0;  // ... and again at full depth
    unsigned fut_pruned = 0;      // moves of explicit nodes skipped by futility pruning (tasks not counted)
    unsigned killer_stores = 0;   // explicit-node cutoffs that changed their ply's killers (tasks not counted)
    unsigned killers_used = 0;    // explicit nodes with a killer among their moves
    unsigned qs_pruned = 0;       // moves of explicit quiescence nodes skipped by delta or SEE pruning
    int nodes_allocated = 0;
    int moves_allocated = 0;
    bool overflow = false;     // a pool was too small (the search is reported aborted)
    double kernel_ms = 0;
    double total_ms = 0;
    int blocks = 0;
    int block_size = 0;
};

std::string describe(const YbwcDeviceStats& s);

class GpuYbwcDevice {
  public:
    GpuYbwcDevice(const DeviceContext& ctx, const GpuSearchOptions& options, const YbwcDeviceLimits& limits = {});
    ~GpuYbwcDevice();
    GpuYbwcDevice(const GpuYbwcDevice&) = delete;
    GpuYbwcDevice& operator=(const GpuYbwcDevice&) = delete;

    // One search of the root at `depth` with the window (alpha, beta), fail-soft.
    FixedDepthResult search(const Position& root, const GameHistory& history, int depth, const YbwcOptions& options,
                            const std::vector<Move>& pv_hint = {}, YbwcDeviceStats* stats = nullptr,
                            int alpha = -kScoreInfinite, int beta = kScoreInfinite);

    // As GpuSearcher: safe from any thread; forwarded by the searching thread.
    void request_stop();
    void clear_stop();
    void set_deadline(std::chrono::steady_clock::time_point deadline);
    // The table (device memory, e.g. a DeviceTT) for feature sets with one.
    void set_tt(const TTDesc& tt);
    // Feature sets with killer moves: the explicit nodes' killers by ply
    // after the last search (for tests); empty for the others.
    std::vector<Killers> explicit_killers() const;

    const GpuSearchOptions& options() const;
    std::string describe() const;

    struct Impl;  // one subclass per board and eval (ybwc_device.cu)

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace cb
