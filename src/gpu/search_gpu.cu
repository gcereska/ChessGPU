#include "gpu/search_gpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "gpu/cuda_check.h"
#include "gpu/kernel_registry.h"
#include "gpu/search_kernels.cuh"
#include "gpu/stop_flag.h"
#include "gpu/variants.h"
#include "host/host_tables.h"
#include "host/profiling.h"

namespace cb {

bool is_compiled(EvalKind e) {
    switch (e) {
        case EvalKind::Material: return CHESSGPU_WITH_MATERIAL_EVAL != 0;
        case EvalKind::Pesto: return CHESSGPU_WITH_PESTO_EVAL != 0;
        case EvalKind::Terms: return CHESSGPU_WITH_TERMS_EVAL != 0;
        case EvalKind::Nnue: return CHESSGPU_WITH_NNUE_EVAL != 0;
    }
    return false;
}

bool is_compiled(OrderingKind o) {
    return o == OrderingKind::Staged ? CHESSGPU_WITH_STAGED_ORDERING != 0 : CHESSGPU_WITH_LIST_ORDERING != 0;
}

bool is_compiled(StackKind s) {
    return s == StackKind::Local ? CHESSGPU_WITH_LOCAL_STACK != 0 : CHESSGPU_WITH_SHARED_STACK != 0;
}

bool is_compiled(FeatureKind f, TTLayoutKind layout) {
    const bool layout_ok = layout == TTLayoutKind::Packed64 ? CHESSGPU_WITH_PACKED64_LAYOUT != 0
                                                            : CHESSGPU_WITH_WIDE128_LAYOUT != 0;
    switch (f) {
#define CB_COMPILED_FEATURES_CASE(Kind, Type, name, flags) \
    case FeatureKind::Kind:                                \
        return CHESSGPU_WITH_FEATURES_##Kind != 0 && (!Type::kTT || layout_ok);
        CB_FEATURE_SETS(CB_COMPILED_FEATURES_CASE)
#undef CB_COMPILED_FEATURES_CASE
    }
    return false;
}

bool is_compiled_for_search(AttacksKind a) {
    switch (a) {
        case AttacksKind::Hq: return CHESSGPU_WITH_HQ_SEARCH_ATTACKS != 0;
        case AttacksKind::Magic: return CHESSGPU_WITH_MAGIC_SEARCH_ATTACKS != 0;
        case AttacksKind::Kogge: return false;
    }
    return false;
}

namespace {

[[noreturn]] void fail_not_compiled(const std::string& what, const char* option) {
    throw std::invalid_argument(what + " is not compiled into this build (CMake option " + option + ")");
}

}  // namespace

void validate_shared(const GpuSearchOptions& o) {
    const auto fail = fail_not_compiled;
    if (!is_compiled(o.board)) fail(std::string("board '") + name_of(o.board) + "'", "CHESSGPU_BOARDS");
    if (!is_compiled(o.eval)) fail(std::string("eval '") + name_of(o.eval) + "'", "CHESSGPU_EVALS");
    if (!is_compiled(o.features, o.tt_layout))
        fail(std::string("feature set '") + name_of(o.features) + "'" +
                 (uses_tt(o.features) ? std::string(" with table layout '") + name_of(o.tt_layout) + "'" : ""),
             "CHESSGPU_FEATURES or CHESSGPU_TT_LAYOUTS");
    if (!is_compiled_for_search(o.attacks))
        fail(std::string("attacks '") + name_of(o.attacks) + "' in the search", "CHESSGPU_SEARCH_ATTACKS");
    validate_prune(o.prune);
    if (o.block_size < 32 || o.block_size > 1024 || o.block_size % 32 != 0)
        throw std::invalid_argument("block size must be a multiple of 32 between 32 and 1024");
}

void validate(const GpuSearchOptions& o) {
    const auto fail = fail_not_compiled;
    validate_shared(o);
    if (!is_compiled(o.model)) fail(std::string("execution model '") + name_of(o.model) + "'", "CHESSGPU_MODELS");
    if (o.model == ExecModel::Thread) {
        if (!is_compiled(o.ordering)) fail(std::string("ordering '") + name_of(o.ordering) + "'", "CHESSGPU_ORDERINGS");
        if (o.ordering == OrderingKind::List && o.stack == StackKind::Shared)
            throw std::invalid_argument("the shared-memory stack needs staged ordering (a move list per ply doesn't fit)");
        if (o.ordering == OrderingKind::Staged && !is_compiled(o.stack))
            fail(std::string("stack '") + name_of(o.stack) + "'", "CHESSGPU_STACKS");
    }
}

namespace {

// Device array that grows when a batch needs more room.
template <class T>
class GrowBuffer {
  public:
    GrowBuffer() = default;
    ~GrowBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    GrowBuffer(const GrowBuffer&) = delete;
    GrowBuffer& operator=(const GrowBuffer&) = delete;

    void reserve(size_t n) {
        if (n <= capacity_) return;
        if (ptr_) CUDA_CHECK(cudaFree(ptr_));
        ptr_ = nullptr;
        const size_t c = std::max(n, capacity_ * 2);
        CUDA_CHECK(cudaMalloc(&ptr_, c * sizeof(T)));
        capacity_ = c;
    }
    T* get() const { return ptr_; }
    size_t capacity() const { return capacity_; }

  private:
    T* ptr_ = nullptr;
    size_t capacity_ = 0;
};

// Page-locked host array that grows as needed. Copies between it and the
// device are truly asynchronous, so a thread waiting for a batch never sits
// inside a blocking copy call while another thread asks the batch to stop.
template <class T>
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void reserve(size_t n) {
        if (n <= capacity_) return;
        if (ptr_) CUDA_CHECK(cudaFreeHost(ptr_));
        ptr_ = nullptr;
        const size_t c = std::max(n, capacity_ * 2);
        CUDA_CHECK(cudaHostAlloc(&ptr_, c * sizeof(T), cudaHostAllocDefault));
        capacity_ = c;
    }
    T* get() const { return ptr_; }
    T& operator[](size_t i) const { return ptr_[i]; }

  private:
    T* ptr_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace

struct GpuSearcher::Impl {
    explicit Impl(const DeviceContext& context, const GpuSearchOptions& opt) : ctx(context), options(opt) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d_next, sizeof(unsigned)));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        if (uses_lmr(opt.features)) {
            const LmrTable table = make_lmr_table(opt.prune);
            CUDA_CHECK(cudaMalloc(&d_lmr, sizeof table));
            CUDA_CHECK(cudaMemcpy(d_lmr, &table, sizeof table, cudaMemcpyHostToDevice));
        }
    }
    virtual ~Impl() {
        cudaFree(d_lmr);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(done);
        cudaFree(d_next);
        cudaStreamDestroy(work);
    }

    virtual std::vector<SearchResult> run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                          size_t history_unchanged, const std::vector<Move>& pv) = 0;

    // Makes the device history equal `history`, sending only what changed.
    void sync_history(const std::vector<HistoryEntry>& history, size_t unchanged) {
        if (history.empty()) return;
        size_t first = std::min(unchanged, history_on_device);
        if (history.size() > d_history.capacity()) {
            d_history.reserve(history.size());  // a new allocation: send everything
            first = 0;
        }
        const size_t count = history.size() - first;
        if (count > 0) {
            h_history.reserve(count);
            std::copy(history.begin() + std::ptrdiff_t(first), history.end(), h_history.get());
            CUDA_CHECK(cudaMemcpyAsync(d_history.get() + first, h_history.get(), count * sizeof(HistoryEntry),
                                       cudaMemcpyHostToDevice, work));
        }
        history_on_device = history.size();
    }

    // Sets the stop flag for the coming kernel.
    void arm_stop_flag() {
        stop_flag.set(stop_requested.load());
        stop_flag.reset_device(work);
    }

    // Waits for the kernel just queued on the work stream. While waiting,
    // forwards a stop request (stop_requested, set by any thread) or a passed
    // deadline to the running kernel through the mapped stop flag.
    void wait_for_kernel() {
        CUDA_CHECK(cudaEventRecord(done, work));
        bool stop_sent = false;
        for (;;) {
            const cudaError_t status = cudaEventQuery(done);
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

    const DeviceContext& ctx;
    GpuSearchOptions options;
    cudaStream_t work = nullptr;  // uploads, kernels, downloads
    HostStopFlag stop_flag;
    unsigned* d_next = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t done = nullptr;
    std::atomic<bool> stop_requested{false};
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    TTDesc tt{};  // feature sets with a TT: the table (set_tt)
    LmrTable* d_lmr = nullptr;  // feature sets with late move reductions: built from options.prune
    GrowBuffer<HistoryEntry> d_history;
    GrowBuffer<Move> d_pv;
    GrowBuffer<gpu::TaskResult> d_results;
    size_t history_on_device = 0;  // entries of d_history that match the last history sent
    GpuBatchStats stats;
    PinnedBuffer<HistoryEntry> h_history;
    PinnedBuffer<Move> h_pv;
    PinnedBuffer<gpu::TaskResult> h_results;
};

namespace {

template <class Board, class Eval>
class ImplT final : public GpuSearcher::Impl {
  public:
    ImplT(const DeviceContext& context, const GpuSearchOptions& opt) : Impl(context, opt) {
        select_kernel();
        if (shared_per_block_ > ctx.info().shared_per_block_optin)
            throw std::invalid_argument("this block size needs " + std::to_string(shared_per_block_) +
                                        " B of shared memory per block; the GPU allows " +
                                        std::to_string(ctx.info().shared_per_block_optin));
        if (shared_per_block_ > 48 * 1024)
            CUDA_CHECK(
                cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_per_block_)));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_, kernel_, options.block_size,
                                                                 shared_per_block_));
        if (blocks_per_sm_ == 0) throw std::runtime_error("the search kernel does not fit on an SM with these settings");
    }

    std::vector<SearchResult> run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                  size_t history_unchanged, const std::vector<Move>& pv) override {
        const Stopwatch total;
        const size_t n = tasks.size();
        std::vector<SearchResult> results(n);
        if (n == 0) return results;
        if (n > 0x7FFFFFFFull) throw std::invalid_argument("too many tasks");
        if (uses_tt(options.features) && tt.buckets == nullptr)
            throw std::invalid_argument("the feature set needs a table (set_tt)");

        host_tasks_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const TaskSpec& g = tasks[i];
            if (g.depth < -kMaxPly || g.depth >= kMaxPly || g.ply < 0 || g.ply > kMaxPly)
                throw std::invalid_argument("task depth or ply out of range");
            if (g.history >= int(history.size())) throw std::invalid_argument("task history index out of range");
            SearchTask<Board>& t = host_tasks_[i];
            t.board = convert_board<Board>(g.board);
            t.alpha = short(std::clamp(g.alpha, -kScoreInfinite, kScoreInfinite));
            t.beta = short(std::clamp(g.beta, -kScoreInfinite, kScoreInfinite));
            t.depth = static_cast<signed char>(g.depth);
            t.ply = u8(g.ply);
            t.flags = g.flags;
            t.unused = 0;
            t.history = g.history;
            t.node_budget = g.node_budget;
        }
        d_tasks_.reserve(n);
        d_results.reserve(n);
        d_pv.reserve(std::max<size_t>(pv.size(), 1));
        h_results.reserve(n);
        h_pv.reserve(std::max<size_t>(pv.size(), 1));
        std::copy(pv.begin(), pv.end(), h_pv.get());
        CUDA_CHECK(cudaMemcpyAsync(d_tasks_.get(), host_tasks_.get(), n * sizeof(SearchTask<Board>),
                                   cudaMemcpyHostToDevice, work));
        sync_history(history, history_unchanged);
        if (!pv.empty())
            CUDA_CHECK(cudaMemcpyAsync(d_pv.get(), h_pv.get(), pv.size() * sizeof(Move), cudaMemcpyHostToDevice, work));
        CUDA_CHECK(cudaMemsetAsync(d_next, 0, sizeof(unsigned), work));
        arm_stop_flag();

        // As many blocks as fit on the GPU, but no more than the tasks need.
        // Thread model: as few working lanes per warp as the batch allows
        // (see search_thread_local_kernel), so small batches get a warp per task.
        const size_t full = size_t(blocks_per_sm_) * size_t(ctx.info().sm_count);
        int lanes = 32;
        size_t workers_per_block = size_t(workers_per_block_);
        if (options.model == ExecModel::Thread) {
            const size_t warps = full * size_t(options.block_size / 32);
            lanes = int(std::clamp<size_t>((n + warps - 1) / warps, 1, 32));
            workers_per_block = size_t(options.block_size / 32) * size_t(lanes);
        }
        const size_t needed = (n + workers_per_block - 1) / workers_per_block;
        const int blocks = int(std::min(full, needed));
        const gpu::StopSignal signal = stop_flag.signal();
        CUDA_CHECK(cudaEventRecord(start, work));
        kernel_<<<blocks, options.block_size, shared_per_block_, work>>>(
            d_tasks_.get(), unsigned(n),
            gpu::AttackSources{ctx.device_tables(),
                               options.attacks == AttacksKind::Magic ? ctx.device_magic_tables() : nullptr},
            ctx.device_search_tables(),
            history.empty() ? nullptr : d_history.get(), pv.empty() ? nullptr : d_pv.get(), int(pv.size()), signal,
            d_next, d_results.get(), lanes, tt, options.prune, d_lmr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, work));
        wait_for_kernel();
        CUDA_CHECK(cudaMemcpyAsync(h_results.get(), d_results.get(), n * sizeof(gpu::TaskResult),
                                   cudaMemcpyDeviceToHost, work));
        CUDA_CHECK(cudaStreamSynchronize(work));

        for (size_t i = 0; i < n; ++i) {
            const gpu::TaskResult& r = h_results[i];
            results[i].score = r.score;
            results[i].best = Move::from_raw(r.best);
            results[i].nodes = r.nodes;
            results[i].aborted = r.aborted != 0;
            results[i].over_budget = r.over_budget != 0;
        }
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        stats.tasks = n;
        stats.blocks = blocks;
        stats.block_size = options.block_size;
        stats.blocks_per_sm = blocks_per_sm_;
        stats.lanes = lanes;
        stats.shared_bytes = shared_per_block_;
        stats.kernel_ms = ms;
        stats.total_ms = total.elapsed_ms();
        return results;
    }

  private:
    void select_kernel() {
        const gpu::SearchKernelChoice<Board> c = gpu::with_features<gpu::SearchKernelChoice<Board>>(
            options, [&]<class Features, class Layout>(gpu::Tag<Features>, gpu::Tag<Layout>) {
                return gpu::search_kernel_for<Board, Eval, Features, Layout>(options);
            });
        if (c.kernel == nullptr) throw std::invalid_argument("this search variant is not compiled in");
        kernel_ = c.kernel;
        shared_per_block_ = c.shared_per_block;
        workers_per_block_ = c.workers_per_block;
    }

    gpu::SearchKernel<Board> kernel_ = nullptr;
    size_t shared_per_block_ = 0;
    int workers_per_block_ = 0;
    int blocks_per_sm_ = 0;
    GrowBuffer<SearchTask<Board>> d_tasks_;
    PinnedBuffer<SearchTask<Board>> host_tasks_;
};

template <class Board>
std::unique_ptr<GpuSearcher::Impl> make_impl_for_board(const DeviceContext& ctx, const GpuSearchOptions& o) {
#if CHESSGPU_WITH_MATERIAL_EVAL
    if (o.eval == EvalKind::Material) return std::make_unique<ImplT<Board, MaterialEval>>(ctx, o);
#endif
#if CHESSGPU_WITH_PESTO_EVAL
    if (o.eval == EvalKind::Pesto) return std::make_unique<ImplT<Board, PestoEval>>(ctx, o);
#endif
#if CHESSGPU_WITH_TERMS_EVAL
    if (o.eval == EvalKind::Terms) return std::make_unique<ImplT<Board, PestoTermsEval>>(ctx, o);
#endif
#if CHESSGPU_WITH_NNUE_EVAL
    if (o.eval == EvalKind::Nnue) return std::make_unique<ImplT<Board, NnueEval>>(ctx, o);
#endif
    throw std::invalid_argument("eval not compiled in");
}

std::unique_ptr<GpuSearcher::Impl> make_impl(const DeviceContext& ctx, const GpuSearchOptions& o) {
    validate(o);
#if CHESSGPU_WITH_STD_BOARD
    if (o.board == BoardKind::Std) return make_impl_for_board<StdBoard>(ctx, o);
#endif
#if CHESSGPU_WITH_QUAD_BOARD
    if (o.board == BoardKind::Quad) return make_impl_for_board<QuadBoard>(ctx, o);
#endif
    throw std::invalid_argument("board not compiled in");
}

}  // namespace

GpuSearcher::GpuSearcher(const DeviceContext& ctx, const GpuSearchOptions& options)
    : impl_(make_impl(ctx, options)) {}

GpuSearcher::~GpuSearcher() = default;

const GpuSearchOptions& GpuSearcher::options() const { return impl_->options; }

const GpuBatchStats& GpuSearcher::last_batch_stats() const { return impl_->stats; }

std::vector<SearchResult> GpuSearcher::run(const std::vector<TaskSpec>& tasks, const std::vector<HistoryEntry>& history,
                                           size_t history_unchanged, const std::vector<Move>& pv) {
    return impl_->run(tasks, history, history_unchanged, pv);
}

std::string describe(const GpuSearchOptions& o) {
    std::string s = std::string(name_of(o.board)) + "/" + name_of(o.eval) + "/" + name_of(o.model);
    if (o.model == ExecModel::Thread) s += std::string("/") + name_of(o.ordering) + "/" + name_of(o.stack);
    s += " block " + std::to_string(o.block_size);
    if (o.attacks != AttacksKind::Hq) s += std::string(" ") + name_of(o.attacks);
    if (uses_tt(o.features)) s += std::string(" ") + name_of(o.features) + " " + name_of(o.tt_layout);
    return s + describe_prune(o.features, o.prune);
}

std::string GpuSearcher::describe() const { return "gpu " + cb::describe(impl_->options); }
EvalKind GpuSearcher::eval_kind() const { return impl_->options.eval; }

void GpuSearcher::request_stop() { impl_->stop_requested.store(true); }
void GpuSearcher::clear_stop() { impl_->stop_requested.store(false); }
void GpuSearcher::set_deadline(std::chrono::steady_clock::time_point deadline) { impl_->deadline = deadline; }
void GpuSearcher::set_tt(const TTDesc& tt) { impl_->tt = tt; }

}  // namespace cb
