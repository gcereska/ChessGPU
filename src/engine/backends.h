#pragma once

// The engine's search backends and their options.

#include <memory>
#include <string>
#include <string_view>

#include "gpu/device_tt.h"
#include "gpu/abdada_device.h"
#include "gpu/search_gpu.h"
#include "gpu/ybwc_device.h"
#include "host/search_backend.h"
#include "host/ybwc.h"

namespace cb {

// Which parallel search runs (see docs/ARCHITECTURE.md, Parallel search on the GPU).
enum class SearchKind {
    Cpu,             // single-threaded CPU search
    GpuIndependent,  // CPU expands the first plies, GPU searches every subtree with a full window
    YbwcHost,        // YBWC: CPU coordinates the explicit nodes in rounds, GPU searches the tasks
    YbwcDevice,      // YBWC entirely on the GPU: one persistent kernel per iteration
    AbdadaCpu,       // ABDADA on CPU threads
    AbdadaDevice,    // ABDADA on the GPU: one persistent kernel of workers per iteration
};

// Does this search run on the GPU (and need a device context)?
constexpr bool needs_gpu(SearchKind k) { return k != SearchKind::Cpu && k != SearchKind::AbdadaCpu; }

const char* name_of(SearchKind k);
bool parse_search_kind(std::string_view text, SearchKind& out);

// The engine's feature set: the last one to win its match against the one
// before (docs/ARCHITECTURE.md, Results). Search options elsewhere default to
// none, the plain search, which tests compare against.
constexpr FeatureKind kEngineFeatures = FeatureKind::TTKillersSee;

// The engine's evaluation, likewise (docs/ARCHITECTURE.md, Evaluation): the NNUE
// with the compiled-in fifth network (src/host/nnue_default.cpp) beat the
// hand-written terms by +64 Elo. Search options elsewhere default to PeSTO.
constexpr EvalKind kEngineEval = EvalKind::Nnue;

inline GpuSearchOptions engine_gpu_defaults() {
    GpuSearchOptions g;
    g.features = kEngineFeatures;
    g.eval = kEngineEval;
    return g;
}

struct EngineOptions {
    SearchKind search = SearchKind::YbwcDevice;  // the fastest (docs/ARCHITECTURE.md, Results)
    // Board, eval, feature set, table layout and pruning settings also apply to the CPU search.
    GpuSearchOptions gpu = engine_gpu_defaults();
    size_t hash_mb = kDefaultHashMb;  // transposition table size (feature sets with one)
    OrderingKind cpu_ordering = OrderingKind::Staged;
    int split_depth = 2;  // GpuIndependent: plies expanded on the CPU
    YbwcOptions ybwc;     // YbwcHost and YbwcDevice
    YbwcDeviceLimits device_limits;  // YbwcDevice
    int threads = 4;                 // AbdadaCpu: workers
    unsigned workers = 2048;         // AbdadaDevice: workers (capped at what fits); lanes per warp as YbwcDevice
    AbdadaParams abdada{};           // AbdadaCpu and AbdadaDevice: coordination settings
    int aspiration_cp = 0;           // aspiration windows' first margin (SearchRequest::aspiration_cp); 0: off
    int fail_low_percent = 0;        // time added after a fail low, % of the hard limit (plan_time); 0: none
};

// Builds the backend. ctx may be null for searches that don't need_gpu. Throws
// std::invalid_argument for a variant that isn't compiled in.
std::unique_ptr<SearchBackend> make_backend(const EngineOptions& options, const DeviceContext* ctx);

}  // namespace cb
