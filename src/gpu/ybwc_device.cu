#include "gpu/ybwc_device.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "gpu/cuda_check.h"
#include "gpu/kernel_registry.h"
#include "gpu/stop_flag.h"
#include "gpu/variants.h"
#include "host/host_tables.h"
#include "host/profiling.h"

namespace cb {

namespace {

template <class T>
T* device_alloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

}  // namespace

std::string describe(const YbwcDeviceStats& s) {
    char text[256];
    std::snprintf(text, sizeof text,
                  "tasks %u (skipped %u, over budget %u) | explicit %u (null moves %u, cutoffs %u; reduced %u, "
                  "searched again %u; futile moves %u; killers stored %u, used %u; quiescence moves pruned %u) "
                  "| nodes %d, moves %d%s | kernel %.1f ms of %.1f ms",
                  s.tasks, s.skipped, s.over_budget, s.explicit_nodes, s.null_tries, s.null_cuts, s.lmr_reductions,
                  s.lmr_researches, s.fut_pruned, s.killer_stores, s.killers_used, s.qs_pruned, s.nodes_allocated,
                  s.moves_allocated, s.overflow ? " OVERFLOW" : "", s.kernel_ms, s.total_ms);
    return text;
}

struct GpuYbwcDevice::Impl {
    using Kernel = void (*)(gpu::YbwcDeviceArgs, int);

    Impl(const DeviceContext& context, const GpuSearchOptions& opt, const YbwcDeviceLimits& lim)
        : ctx(context), options(opt), limits(lim) {
        validate(opt);
        if (lim.max_nodes < 16 || lim.max_moves < 1024 || lim.lanes < 1 || lim.lanes > 32)
            throw std::invalid_argument("bad YBWC device limits");
        CUDA_CHECK(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        queue_capacity = unsigned(lim.max_nodes) + (1u << 20);
        d_nodes = device_alloc<gpu::YNode>(size_t(lim.max_nodes));
        d_moves = device_alloc<Move>(size_t(lim.max_moves));
        d_queue = device_alloc<int>(queue_capacity);
        d_ready = device_alloc<unsigned>(queue_capacity);
        d_counters = device_alloc<unsigned long long>(kCounterWords);
        d_pv = device_alloc<Move>(kMaxPly + 1);
        if (uses_lmr(opt.features)) {
            const LmrTable table = make_lmr_table(opt.prune);
            d_lmr = device_alloc<LmrTable>(1);
            CUDA_CHECK(cudaMemcpy(d_lmr, &table, sizeof table, cudaMemcpyHostToDevice));
        }
        if (uses_killers(opt.features)) d_killers = device_alloc<u32>(kMaxPly + 1);
    }
    virtual ~Impl() {
        cudaFree(d_killers);
        cudaFree(d_lmr);
        cudaFree(d_history);
        cudaFree(d_pv);
        cudaFree(d_counters);
        cudaFree(d_ready);
        cudaFree(d_queue);
        cudaFree(d_moves);
        cudaFree(d_nodes);
        if (done_event_) cudaEventDestroy(done_event_);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(work);
    }

    void ensure_history(size_t entries) {
        if (entries <= history_capacity) return;
        cudaFree(d_history);
        d_history = device_alloc<HistoryEntry>(entries);
        history_capacity = entries;
    }

    FixedDepthResult search(const Position& root, const GameHistory& history, int depth, const YbwcOptions& yo,
                            const std::vector<Move>& pv_hint, YbwcDeviceStats* out_stats, int alpha, int beta) {
        const Stopwatch total;
        if (depth < 0 || depth >= kMaxPly) throw std::invalid_argument("search depth out of range");
        if (alpha < -kScoreInfinite || beta > kScoreInfinite || alpha >= beta)
            throw std::invalid_argument("search window out of range");
        if (uses_tt(options.features) && tt.buckets == nullptr)
            throw std::invalid_argument("the feature set needs a table (set_tt)");
        const int game_count = int(history.entries.size());
        ensure_history(size_t(game_count) + size_t(limits.max_nodes));

        // Counters, queue and pools start empty; the root is node 0, queued.
        gpu::YNode r{};
        r.board = to_board<QuadBoard>(root);
        r.parent = -1;
        r.depth = depth;
        r.alpha = alpha;
        r.beta = beta;
        r.parent_alpha = -kScoreInfinite;  // unused: the root has no parent
        r.best = -kScoreInfinite;
        r.best_child = -1;
        r.on_pv = 1;
        r.task = (depth > yo.leaf_depth && 0 < yo.max_explicit_ply) ? 0 : 1;
        r.state = r.task ? gpu::kYQueued : gpu::kYIdle;
        r.type = gpu::kYPv;
        CUDA_CHECK(cudaMemsetAsync(d_counters, 0, kCounterWords * sizeof(unsigned long long), work));
        CUDA_CHECK(cudaMemsetAsync(d_ready, 0, queue_capacity * sizeof(unsigned), work));
        if (d_killers) CUDA_CHECK(cudaMemsetAsync(d_killers, 0, (kMaxPly + 1) * sizeof(u32), work));
        h_init.root = r;
        h_init.node_count = 1;
        h_init.tail = 1;
        h_init.queue0 = 0;
        h_init.ready0 = 1;
        CUDA_CHECK(cudaMemcpyAsync(d_nodes, &h_init.root, sizeof(gpu::YNode), cudaMemcpyHostToDevice, work));
        CUDA_CHECK(cudaMemcpyAsync(node_count_ptr(), &h_init.node_count, sizeof(int), cudaMemcpyHostToDevice, work));
        CUDA_CHECK(cudaMemcpyAsync(tail_ptr(), &h_init.tail, sizeof(unsigned), cudaMemcpyHostToDevice, work));
        CUDA_CHECK(cudaMemcpyAsync(d_queue, &h_init.queue0, sizeof(int), cudaMemcpyHostToDevice, work));
        CUDA_CHECK(cudaMemcpyAsync(d_ready, &h_init.ready0, sizeof(unsigned), cudaMemcpyHostToDevice, work));
        if (game_count > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_history, history.entries.data(), size_t(game_count) * sizeof(HistoryEntry),
                                       cudaMemcpyHostToDevice, work));
        const int pv_length = int(std::min<size_t>(pv_hint.size(), kMaxPly + 1));
        if (pv_length > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_pv, pv_hint.data(), size_t(pv_length) * sizeof(Move), cudaMemcpyHostToDevice,
                                       work));
        stop_flag.set(stop_requested.load());
        stop_flag.reset_device(work);

        gpu::YbwcDeviceArgs a{};
        a.attack_tables = ctx.device_tables();
        a.magic_tables = options.attacks == AttacksKind::Magic ? ctx.device_magic_tables() : nullptr;
        a.search_tables = ctx.device_search_tables();
        a.pv = pv_length > 0 ? d_pv : nullptr;
        a.pv_length = pv_length;
        a.history = d_history;
        a.game_count = game_count;
        a.root_repeated = history.root_repeated ? 1 : 0;
        a.leaf_depth = yo.leaf_depth;
        a.max_explicit_ply = yo.max_explicit_ply;
        a.node_budget = yo.node_budget;
        a.cut_waves = yo.cut_waves ? 1 : 0;
        a.nodes = d_nodes;
        a.node_capacity = limits.max_nodes;
        a.node_count = node_count_ptr();
        a.moves = d_moves;
        a.move_capacity = limits.max_moves;
        a.move_count = move_count_ptr();
        a.queue = d_queue;
        a.ready = d_ready;
        a.queue_capacity = queue_capacity;
        a.head = head_ptr();
        a.tail = tail_ptr();
        a.done = done_ptr();
        a.signal = stop_flag.signal();
        a.nodes_searched = d_counters + kNodesSearched;
        a.explicit_nodes = word_ptr(kExplicit);
        a.tasks = word_ptr(kTasks);
        a.skipped = word_ptr(kSkipped);
        a.over_budget = word_ptr(kOverBudget);
        a.overflow = reinterpret_cast<int*>(word_ptr(kOverflow));
        a.null_tries = word_ptr(kNullTries);
        a.null_cuts = word_ptr(kNullCuts);
        a.lmr_reductions = word_ptr(kLmrReductions);
        a.lmr_researches = word_ptr(kLmrResearches);
        a.fut_pruned = word_ptr(kFutPruned);
        a.killer_stores = word_ptr(kKillerStores);
        a.killers_used = word_ptr(kKillersUsed);
        a.qs_pruned = word_ptr(kQsPruned);
        a.killers = d_killers;
        a.tt = tt;
        a.prune = options.prune;
        a.lmr = d_lmr;

        CUDA_CHECK(cudaEventRecord(start, work));
        kernel_<<<blocks_, options.block_size, shared_per_block_, work>>>(a, limits.lanes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, work));
        wait_for_kernel();

        // Results: counters, then the best-child chain from the root.
        unsigned long long counters[kCounterWords];
        CUDA_CHECK(cudaMemcpy(counters, d_counters, sizeof counters, cudaMemcpyDeviceToHost));
        const auto word = [&](int i) { return reinterpret_cast<const unsigned*>(counters)[i]; };
        const int done = int(word(kDone));
        YbwcDeviceStats s;
        s.explicit_nodes = word(kExplicit);
        s.tasks = word(kTasks);
        s.skipped = word(kSkipped);
        s.over_budget = word(kOverBudget);
        s.null_tries = word(kNullTries);
        s.null_cuts = word(kNullCuts);
        s.lmr_reductions = word(kLmrReductions);
        s.lmr_researches = word(kLmrResearches);
        s.fut_pruned = word(kFutPruned);
        s.killer_stores = word(kKillerStores);
        s.killers_used = word(kKillersUsed);
        s.qs_pruned = word(kQsPruned);
        s.overflow = word(kOverflow) != 0;
        s.nodes_allocated = std::min(int(word(kNodeCount)), limits.max_nodes);
        s.moves_allocated = std::min(int(word(kMoveCount)), limits.max_moves);
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        s.kernel_ms = ms;
        s.blocks = blocks_;
        s.block_size = options.block_size;

        FixedDepthResult result;
        result.search.nodes = counters[kNodesSearched] + s.explicit_nodes;
        result.search.aborted = done != 1;
        if (!result.search.aborted) {
            gpu::YNode node;
            CUDA_CHECK(cudaMemcpy(&node, d_nodes, sizeof node, cudaMemcpyDeviceToHost));
            result.search.score = node.value;
            for (int guard = 0; guard <= kMaxPly; ++guard) {
                if (node.task) {
                    if (node.task_best != 0) result.pv.push_back(Move::from_raw(node.task_best));
                    break;
                }
                if (node.best_child < 0) break;
                CUDA_CHECK(cudaMemcpy(&node, d_nodes + node.best_child, sizeof node, cudaMemcpyDeviceToHost));
                result.pv.push_back(Move::from_raw(node.move));
            }
            result.search.best = result.pv.empty() ? Move() : result.pv.front();
        }
        s.total_ms = total.elapsed_ms();
        if (out_stats) *out_stats = s;
        return result;
    }

    std::vector<Killers> explicit_killers() const {
        if (!d_killers) return {};
        std::vector<u32> words(kMaxPly + 1);
        CUDA_CHECK(cudaMemcpy(words.data(), d_killers, words.size() * sizeof(u32), cudaMemcpyDeviceToHost));
        std::vector<Killers> k;
        for (const u32 w : words) k.push_back(unpack_killers(w));
        return k;
    }

    void wait_for_kernel() {
        CUDA_CHECK(cudaEventRecord(done_event(), work));
        bool stop_sent = false;
        for (;;) {
            const cudaError_t status = cudaEventQuery(done_event());
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) CUDA_CHECK(status);
            if (!stop_sent && (stop_requested.load(std::memory_order_relaxed) ||
                               std::chrono::steady_clock::now() >= deadline)) {
                stop_flag.set(true);
                stop_sent = true;
            }
            std::this_thread::yield();
        }
    }

    cudaEvent_t done_event() {
        if (!done_event_) CUDA_CHECK(cudaEventCreateWithFlags(&done_event_, cudaEventDisableTiming));
        return done_event_;
    }

    // Counter words (unsigned) inside d_counters, after the 64-bit node total.
    enum : int {
        kNodesSearched = 0,  // 64-bit word 0
        kNodeCount = 2,      // 32-bit word indices from here on
        kMoveCount,
        kHead,
        kTail,
        kDone,
        kExplicit,
        kTasks,
        kSkipped,
        kOverBudget,
        kOverflow,
        kNullTries,
        kNullCuts,
        kLmrReductions,
        kLmrResearches,
        kFutPruned,
        kKillerStores,
        kKillersUsed,
        kQsPruned,
    };
    static constexpr int kCounterWords = 10;  // 64-bit words: 20 32-bit slots
    static_assert(kQsPruned < 2 * kCounterWords, "counter words full");
    unsigned* word_ptr(int i) const { return reinterpret_cast<unsigned*>(d_counters) + i; }
    int* node_count_ptr() const { return reinterpret_cast<int*>(word_ptr(kNodeCount)); }
    int* move_count_ptr() const { return reinterpret_cast<int*>(word_ptr(kMoveCount)); }
    unsigned* head_ptr() const { return word_ptr(kHead); }
    unsigned* tail_ptr() const { return word_ptr(kTail); }
    int* done_ptr() const { return reinterpret_cast<int*>(word_ptr(kDone)); }

    const DeviceContext& ctx;
    GpuSearchOptions options;
    YbwcDeviceLimits limits;
    cudaStream_t work = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t done_event_ = nullptr;
    HostStopFlag stop_flag;
    std::atomic<bool> stop_requested{false};
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    TTDesc tt{};  // feature sets with a TT: the table (set_tt)
    unsigned queue_capacity = 0;
    gpu::YNode* d_nodes = nullptr;
    Move* d_moves = nullptr;
    int* d_queue = nullptr;
    unsigned* d_ready = nullptr;
    unsigned long long* d_counters = nullptr;
    Move* d_pv = nullptr;
    LmrTable* d_lmr = nullptr;  // feature sets with late move reductions: built from options.prune
    u32* d_killers = nullptr;   // feature sets with killer moves: explicit nodes' killers by ply
    HistoryEntry* d_history = nullptr;
    size_t history_capacity = 0;
    struct Init {
        gpu::YNode root;
        int node_count;
        unsigned tail;
        int queue0;
        unsigned ready0;
    } h_init{};

  protected:
    Kernel kernel_ = nullptr;
    size_t shared_per_block_ = 0;
    int blocks_ = 0;
};

namespace {

template <class Board, class Eval>
class YImplT final : public GpuYbwcDevice::Impl {
  public:
    YImplT(const DeviceContext& context, const GpuSearchOptions& opt, const YbwcDeviceLimits& lim)
        : Impl(context, opt, lim) {
        select_kernel();
        if (shared_per_block_ > ctx.info().shared_per_block_optin)
            throw std::invalid_argument("this block size needs too much shared memory per block");
        if (shared_per_block_ > 48 * 1024)
            CUDA_CHECK(
                cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_per_block_)));
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_, options.block_size,
                                                                 shared_per_block_));
        if (blocks_per_sm == 0) throw std::runtime_error("the YBWC kernel does not fit on an SM with these settings");
        blocks_ = blocks_per_sm * ctx.info().sm_count;
    }

  private:
    void select_kernel() {
        const gpu::YbwcKernelChoice c =
            gpu::with_features<gpu::YbwcKernelChoice>(options, [&]<class Features, class Layout>(gpu::Tag<Features>, gpu::Tag<Layout>) {
                return gpu::ybwc_kernel_for<Board, Eval, Features, Layout>(options);
            });
        if (c.kernel == nullptr) throw std::invalid_argument("this search variant is not compiled in");
        kernel_ = c.kernel;
        shared_per_block_ = c.shared_per_block;
    }
};

template <class Board>
std::unique_ptr<GpuYbwcDevice::Impl> make_y_impl_for_board(const DeviceContext& ctx, const GpuSearchOptions& o,
                                                          const YbwcDeviceLimits& l) {
#if CHESSGPU_WITH_MATERIAL_EVAL
    if (o.eval == EvalKind::Material) return std::make_unique<YImplT<Board, MaterialEval>>(ctx, o, l);
#endif
#if CHESSGPU_WITH_PESTO_EVAL
    if (o.eval == EvalKind::Pesto) return std::make_unique<YImplT<Board, PestoEval>>(ctx, o, l);
#endif
#if CHESSGPU_WITH_TERMS_EVAL
    if (o.eval == EvalKind::Terms) return std::make_unique<YImplT<Board, PestoTermsEval>>(ctx, o, l);
#endif
#if CHESSGPU_WITH_NNUE_EVAL
    if (o.eval == EvalKind::Nnue) return std::make_unique<YImplT<Board, NnueEval>>(ctx, o, l);
#endif
    throw std::invalid_argument("eval not compiled in");
}

std::unique_ptr<GpuYbwcDevice::Impl> make_y_impl(const DeviceContext& ctx, const GpuSearchOptions& o,
                                                 const YbwcDeviceLimits& l) {
    validate(o);
#if CHESSGPU_WITH_STD_BOARD
    if (o.board == BoardKind::Std) return make_y_impl_for_board<StdBoard>(ctx, o, l);
#endif
#if CHESSGPU_WITH_QUAD_BOARD
    if (o.board == BoardKind::Quad) return make_y_impl_for_board<QuadBoard>(ctx, o, l);
#endif
    throw std::invalid_argument("board not compiled in");
}

}  // namespace

GpuYbwcDevice::GpuYbwcDevice(const DeviceContext& ctx, const GpuSearchOptions& options, const YbwcDeviceLimits& limits)
    : impl_(make_y_impl(ctx, options, limits)) {}

GpuYbwcDevice::~GpuYbwcDevice() = default;

FixedDepthResult GpuYbwcDevice::search(const Position& root, const GameHistory& history, int depth,
                                       const YbwcOptions& options, const std::vector<Move>& pv_hint,
                                       YbwcDeviceStats* stats, int alpha, int beta) {
    return impl_->search(root, history, depth, options, pv_hint, stats, alpha, beta);
}

void GpuYbwcDevice::request_stop() { impl_->stop_requested.store(true); }
void GpuYbwcDevice::clear_stop() { impl_->stop_requested.store(false); }
void GpuYbwcDevice::set_deadline(std::chrono::steady_clock::time_point deadline) { impl_->deadline = deadline; }
void GpuYbwcDevice::set_tt(const TTDesc& tt) { impl_->tt = tt; }
std::vector<Killers> GpuYbwcDevice::explicit_killers() const { return impl_->explicit_killers(); }
const GpuSearchOptions& GpuYbwcDevice::options() const { return impl_->options; }
std::string GpuYbwcDevice::describe() const { return "gpu " + cb::describe(impl_->options); }

}  // namespace cb
