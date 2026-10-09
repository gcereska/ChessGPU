#pragma once

// GPU search: runs batches of independent search tasks (a TaskRunner); the
// parallel searches built on them are in host/ (split_search, ybwc).

#include <chrono>
#include <memory>
#include <vector>

#include "core/search.h"
#include "gpu/device_context.h"
#include "gpu/perft_gpu.h"
#include "host/game_history.h"
#include "host/task_runner.h"
#include "host/variants.h"

namespace cb {

// GPU search variants compiled into this build (CMake options
// CHESSGPU_EVALS, CHESSGPU_ORDERINGS, CHESSGPU_STACKS, CHESSGPU_FEATURES and
// CHESSGPU_TT_LAYOUTS; boards and models as for perft).
bool is_compiled(EvalKind e);
bool is_compiled(OrderingKind o);
bool is_compiled(StackKind s);
bool is_compiled(FeatureKind f, TTLayoutKind layout);

struct GpuSearchOptions {
    BoardKind board = BoardKind::Quad;
    ExecModel model = ExecModel::Thread;
    EvalKind eval = EvalKind::Pesto;
    OrderingKind ordering = OrderingKind::Staged;  // thread model only
    StackKind stack = StackKind::Local;            // thread model with staged ordering only
    FeatureKind features = FeatureKind::None;
    TTLayoutKind tt_layout = TTLayoutKind::Wide128;   // feature sets with a TT (11% faster to depth than packed64)
    PruneParams prune{};                              // settings of the feature set's pruning techniques
    AttacksKind attacks = AttacksKind::Hq;            // sliding-piece attacks in the kernels (hq or magic)
    int block_size = 128;
};

// Sliding-attack variants compiled into the search kernels (CMake option
// CHESSGPU_SEARCH_ATTACKS; the perft kernels have their own list).
bool is_compiled_for_search(AttacksKind a);

// Checks that the variant combination is compiled in and valid; throws
// std::invalid_argument naming the problem otherwise.
void validate(const GpuSearchOptions& options);
// The part every GPU search needs (board, eval, feature set and table layout,
// attacks, pruning settings, block size), without the task kernels'
// execution model, ordering and stack: ABDADA's kernel has its own.
void validate_shared(const GpuSearchOptions& options);

struct GpuBatchStats {
    size_t tasks = 0;
    int blocks = 0;
    int block_size = 0;
    int blocks_per_sm = 0;
    size_t shared_bytes = 0;  // dynamic shared memory per block
    int lanes = 32;           // thread model: working threads per warp
    double kernel_ms = 0;     // GPU time of the kernel (CUDA events)
    double total_ms = 0;      // wall time including uploads and downloads
};

// Runs search tasks on the GPU. Keeps its device buffers (and the history,
// sent incrementally) between batches. Not thread-safe, except
// request_stop() and clear_stop(), which only set a flag that the thread
// running the batch forwards to the GPU.
class GpuSearcher final : public TaskRunner {
  public:
    GpuSearcher(const DeviceContext& ctx, const GpuSearchOptions& options);
    ~GpuSearcher() override;
    GpuSearcher(const GpuSearcher&) = delete;
    GpuSearcher& operator=(const GpuSearcher&) = delete;

    const GpuSearchOptions& options() const;
    // Statistics of the last batch.
    const GpuBatchStats& last_batch_stats() const;

    std::vector<SearchResult> run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                  size_t history_unchanged, const std::vector<Move>& pv) override;
    void request_stop() override;
    void clear_stop() override;
    void set_deadline(std::chrono::steady_clock::time_point deadline) override;
    std::string describe() const override;
    // The table (device memory, e.g. a DeviceTT) for feature sets with one.
    void set_tt(const TTDesc& tt) override;
    EvalKind eval_kind() const override;
    double last_device_ms() const override { return last_batch_stats().kernel_ms; }

    struct Impl;  // implementation (search_gpu.cu), one subclass per board and eval

  private:
    std::unique_ptr<Impl> impl_;
};

// "quad/pesto/thread/staged/local block 128", plus " tt wide128" with a table.
std::string describe(const GpuSearchOptions& options);

}  // namespace cb
