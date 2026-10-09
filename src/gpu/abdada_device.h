#pragma once

// ABDADA on the GPU (docs/ARCHITECTURE.md, Parallel search on the GPU):
// `workers` threads of one persistent kernel search the root together,
// coordinating through the transposition table and a table of positions
// being searched (core/abdada.h). The first to finish answers. Thread model,
// list ordering. Nondeterministic with more than one worker: the score is an
// alpha-beta score of the root (without a table, exactly the sequential one).

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "gpu/device_context.h"
#include "gpu/search_gpu.h"
#include "host/game_history.h"
#include "host/search_cpu.h"

namespace cb {

struct AbdadaDeviceOptions {
    unsigned workers = 2048;  // capped at the workers resident on the GPU at once
    int lanes = 1;            // workers per warp (thread model): 1 to 32
    AbdadaParams params{};    // coordination settings (core/abdada.h)
};

struct AbdadaDeviceStats {
    unsigned workers = 0;  // after the cap
    int lanes = 0;
    int blocks = 0;
    unsigned started = 0;            // workers that ran (all of them)
    unsigned finished = 0;           // workers that searched the root to the end before being stopped
    unsigned long long id_sum = 0;   // sum of worker + 1 over the workers that ran (each number once)
    unsigned long long nodes = 0;     // all workers
    unsigned long long deferred = 0;  // moves put back because another worker was searching them
    double kernel_ms = 0;
    double total_ms = 0;
};

std::string describe(const AbdadaDeviceStats& s);

class GpuAbdadaDevice {
  public:
    // options: board, eval, feature set, table layout, attacks and pruning
    // settings (the model is the thread model, the ordering a move list).
    GpuAbdadaDevice(const DeviceContext& ctx, const GpuSearchOptions& options, int busy_bits = 20);
    ~GpuAbdadaDevice();
    GpuAbdadaDevice(const GpuAbdadaDevice&) = delete;
    GpuAbdadaDevice& operator=(const GpuAbdadaDevice&) = delete;

    // One search of the root at `depth` with the window (alpha, beta), fail-soft.
    // The PV is the best move only (the workers keep no PV table; deeper
    // moves come from the transposition table in the next iteration).
    FixedDepthResult search(const Position& root, const GameHistory& history, int depth,
                            const AbdadaDeviceOptions& abdada, const std::vector<Move>& pv_hint = {},
                            AbdadaDeviceStats* stats = nullptr, int alpha = -kScoreInfinite,
                            int beta = kScoreInfinite);
    // Workers resident on the GPU at once with `lanes` per warp.
    unsigned max_workers(int lanes) const;
    // Positions claimed in the busy table now (tests: a search that ends
    // without stopping a worker releases every claim).
    size_t claims() const;

    // As GpuSearcher: safe from any thread; forwarded by the searching thread.
    void request_stop();
    void clear_stop();
    void set_deadline(std::chrono::steady_clock::time_point deadline);
    // The table (device memory, e.g. a DeviceTT) for feature sets with one.
    void set_tt(const TTDesc& tt);

    const GpuSearchOptions& options() const;
    std::string describe() const;

    struct Impl;  // one subclass per board and eval (abdada_device.cu)

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace cb
