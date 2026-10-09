#pragma once

// Batches of independent search tasks: what parallel search coordinators
// (independent subtrees, YBWC) hand out. The GPU runs them (GpuSearcher); a
// CPU runner gives the same results, so coordinators can be tested and
// debugged on the CPU.

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "core/search.h"
#include "host/search_cpu.h"

namespace cb {

// A search task. The board is always a QuadBoard here; runners convert it to
// the board their kernels use.
struct TaskSpec {
    QuadBoard board;
    int alpha = -kScoreInfinite;
    int beta = kScoreInfinite;
    int depth = 0;
    int ply = 0;
    u8 flags = 0;      // kTaskOnPv, kTaskRootRepeated, kTaskAfterNull
    int history = -1;  // entry of the task root's parent in the history
    u32 node_budget = 0;  // stop (result over_budget) after about this many nodes; 0: none
};

class TaskRunner {
  public:
    virtual ~TaskRunner() = default;

    // Searches every task on its own; results are in task order. `history`
    // holds the entries the tasks link to. Its first `history_unchanged`
    // entries are the same as in the previous call, so a runner that keeps a
    // copy (on the GPU) needn't send them again. `pv` is the previous
    // iteration's PV indexed by ply.
    virtual std::vector<SearchResult> run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                          size_t history_unchanged, const std::vector<Move>& pv) = 0;

    // Makes the running batch and later ones stop early (their results come
    // back aborted) until clear_stop(). Safe to call from any thread.
    virtual void request_stop() = 0;
    virtual void clear_stop() = 0;
    // Batches running past this time stop early. time_point::max(): none.
    virtual void set_deadline(std::chrono::steady_clock::time_point deadline) = 0;

    virtual std::string describe() const = 0;
    // The transposition table for the next runs (feature sets with one): the
    // searcher's table and its current generation. Ignored without the feature.
    virtual void set_tt(const TTDesc&) {}
    // The evaluation the tasks use (coordinators evaluate explicit nodes the same way).
    virtual EvalKind eval_kind() const = 0;
    // Time the device spent searching in the last run (0 for the CPU).
    virtual double last_device_ms() const { return 0; }
};

// Runs tasks one after another with the CPU search.
class CpuTaskRunner final : public TaskRunner {
  public:
    explicit CpuTaskRunner(const CpuSearchOptions& options) : options_(options) {}

    std::vector<SearchResult> run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                  size_t history_unchanged, const std::vector<Move>& pv) override;
    void request_stop() override { stop_.store(true); }
    void clear_stop() override { stop_.store(false); }
    void set_deadline(std::chrono::steady_clock::time_point deadline) override { deadline_ = deadline; }
    std::string describe() const override;
    void set_tt(const TTDesc& tt) override { options_.tt = tt; }
    EvalKind eval_kind() const override { return options_.eval; }

  private:
    CpuSearchOptions options_;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point deadline_ = std::chrono::steady_clock::time_point::max();
};

}  // namespace cb
