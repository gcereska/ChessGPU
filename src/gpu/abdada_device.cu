#include "gpu/abdada_device.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "gpu/cuda_check.h"
#include "gpu/kernel_registry.h"
#include "gpu/stop_flag.h"
#include "gpu/variants.h"
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

std::string describe(const AbdadaDeviceStats& s) {
    char text[256];
    std::snprintf(text, sizeof text,
                  "workers %u (%d per warp, %d blocks; started %u, finished %u) | nodes %llu, deferred %llu | kernel "
                  "%.1f ms of %.1f ms",
                  s.workers, s.lanes, s.blocks, s.started, s.finished, s.nodes, s.deferred, s.kernel_ms, s.total_ms);
    return text;
}

struct GpuAbdadaDevice::Impl {
    Impl(const DeviceContext& context, const GpuSearchOptions& opt, int busy_bits)
        : ctx(context), options(opt) {
        // Its kernel is always the thread model with list ordering, compiled
        // whatever CHESSGPU_MODELS and CHESSGPU_ORDERINGS say.
        validate_shared(opt);
        if (busy_bits < 10 || busy_bits > 30) throw std::invalid_argument("ABDADA busy table bits must be 10 to 30");
        if (opt.block_size < 32 || opt.block_size % 32 != 0)
            throw std::invalid_argument("ABDADA block size must be a multiple of 32");
        busy_slots = size_t(1) << busy_bits;
        CUDA_CHECK(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
        d_busy = device_alloc<u64>(busy_slots);
        d_words = device_alloc<Words>(1);
        d_pv = device_alloc<Move>(kMaxPly + 1);
        if (uses_lmr(opt.features)) {
            const LmrTable table = make_lmr_table(opt.prune);
            d_lmr = device_alloc<LmrTable>(1);
            CUDA_CHECK(cudaMemcpy(d_lmr, &table, sizeof table, cudaMemcpyHostToDevice));
        }
    }
    virtual ~Impl() {
        cudaFree(d_lmr);
        cudaFree(d_history);
        cudaFree(d_pv);
        cudaFree(d_words);
        cudaFree(d_busy);
        cudaEventDestroy(done_event);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(work);
    }

    // Per board: uploads the root task and launches the kernel on `work`.
    virtual void launch(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                        int blocks, const gpu::AbdadaDeviceArgs& a) = 0;

    unsigned max_workers(int lanes) const {
        if (lanes < 1 || lanes > 32) throw std::invalid_argument("ABDADA lanes must be 1 to 32");
        return unsigned(resident_blocks) * unsigned(options.block_size / 32) * unsigned(lanes);
    }

    FixedDepthResult search(const Position& root, const GameHistory& history, int depth,
                            const AbdadaDeviceOptions& ao, const std::vector<Move>& pv_hint,
                            AbdadaDeviceStats* out_stats, int alpha, int beta) {
        const Stopwatch total;
        if (depth < 0 || depth >= kMaxPly) throw std::invalid_argument("search depth out of range");
        if (alpha < -kScoreInfinite || beta > kScoreInfinite || alpha >= beta)
            throw std::invalid_argument("search window out of range");
        if (uses_tt(options.features) && tt.buckets == nullptr)
            throw std::invalid_argument("the feature set needs a table (set_tt)");
        if (ao.workers < 1) throw std::invalid_argument("ABDADA needs at least one worker");
        validate_abdada(ao.params);
        const unsigned workers = std::min(ao.workers, max_workers(ao.lanes));
        const unsigned warps = (workers + unsigned(ao.lanes) - 1) / unsigned(ao.lanes);
        const int blocks = int((size_t(warps) * 32 + size_t(options.block_size) - 1) / size_t(options.block_size));

        const size_t game_count = history.entries.size();
        if (game_count > history_capacity) {
            cudaFree(d_history);
            d_history = device_alloc<HistoryEntry>(game_count);
            history_capacity = game_count;
        }
        if (game_count > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_history, history.entries.data(), game_count * sizeof(HistoryEntry),
                                       cudaMemcpyHostToDevice, work));
        const int pv_length = int(std::min<size_t>(pv_hint.size(), kMaxPly + 1));
        if (pv_length > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_pv, pv_hint.data(), size_t(pv_length) * sizeof(Move), cudaMemcpyHostToDevice,
                                       work));
        // Claims of stopped workers stay behind: every search starts empty.
        CUDA_CHECK(cudaMemsetAsync(d_busy, 0, busy_slots * sizeof(u64), work));
        CUDA_CHECK(cudaMemsetAsync(d_words, 0, sizeof(Words), work));
        stop_flag.set(stop_requested.load());
        stop_flag.reset_device(work);

        gpu::AbdadaDeviceArgs a{};
        a.sources = gpu::AttackSources{ctx.device_tables(),
                                       options.attacks == AttacksKind::Magic ? ctx.device_magic_tables() : nullptr};
        a.search_tables = ctx.device_search_tables();
        a.history = game_count > 0 ? d_history : nullptr;
        a.pv = pv_length > 0 ? d_pv : nullptr;
        a.pv_length = pv_length;
        a.tt = tt;
        a.prune = options.prune;
        a.lmr = d_lmr;
        a.busy = d_busy;
        a.busy_mask = busy_slots - 1;
        a.params = ao.params;
        a.workers = workers;
        a.lanes = ao.lanes;
        a.done = &d_words->done;
        a.result = &d_words->result;
        a.nodes = &d_words->nodes;
        a.deferred = &d_words->deferred;
        a.started = &d_words->started;
        a.finished = &d_words->finished;
        a.id_sum = &d_words->id_sum;
        a.signal = stop_flag.signal();

        CUDA_CHECK(cudaEventRecord(start, work));
        launch(root, history, depth, alpha, beta, blocks, a);
        CUDA_CHECK(cudaEventRecord(stop, work));
        wait_for_kernel();

        Words w;
        CUDA_CHECK(cudaMemcpy(&w, d_words, sizeof w, cudaMemcpyDeviceToHost));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        FixedDepthResult result;
        result.search.nodes = w.nodes;
        result.search.aborted = w.done != 1;
        if (!result.search.aborted) {
            result.search.score = w.result.score;
            result.search.best = Move::from_raw(w.result.best);
            if (!result.search.best.is_none()) result.pv.push_back(result.search.best);
        }
        if (out_stats) {
            AbdadaDeviceStats s;
            s.workers = workers;
            s.lanes = ao.lanes;
            s.blocks = blocks;
            s.started = w.started;
            s.finished = w.finished;
            s.id_sum = w.id_sum;
            s.nodes = w.nodes;
            s.deferred = w.deferred;
            s.kernel_ms = ms;
            s.total_ms = total.elapsed_ms();
            *out_stats = s;
        }
        return result;
    }

    void wait_for_kernel() {
        CUDA_CHECK(cudaEventRecord(done_event, work));
        bool stop_sent = false;
        for (;;) {
            const cudaError_t status = cudaEventQuery(done_event);
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

    // Written by the kernel; zeroed before each search.
    struct Words {
        gpu::TaskResult result;
        unsigned long long nodes;
        unsigned long long deferred;
        unsigned long long id_sum;
        unsigned started;
        unsigned finished;
        int done;
    };

    size_t claims() const {
        std::vector<u64> slots(busy_slots);
        CUDA_CHECK(cudaMemcpy(slots.data(), d_busy, busy_slots * sizeof(u64), cudaMemcpyDeviceToHost));
        return size_t(std::count_if(slots.begin(), slots.end(), [](u64 s) { return s != 0; }));
    }

    const DeviceContext& ctx;
    GpuSearchOptions options;
    size_t busy_slots = 0;
    cudaStream_t work = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t done_event = nullptr;
    HostStopFlag stop_flag;
    std::atomic<bool> stop_requested{false};
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    TTDesc tt{};
    u64* d_busy = nullptr;
    Words* d_words = nullptr;
    Move* d_pv = nullptr;
    LmrTable* d_lmr = nullptr;
    HistoryEntry* d_history = nullptr;
    size_t history_capacity = 0;
    int resident_blocks = 0;  // blocks of the kernel resident on the GPU at once
};

namespace {

template <class Board, class Eval>
class AImplT final : public GpuAbdadaDevice::Impl {
  public:
    AImplT(const DeviceContext& context, const GpuSearchOptions& opt, int busy_bits) : Impl(context, opt, busy_bits) {
        kernel_ = gpu::with_features<gpu::AbdadaKernel<Board>>(
            options, [&]<class Features, class Layout>(gpu::Tag<Features>, gpu::Tag<Layout>) {
                return gpu::abdada_kernel_for<Board, Eval, Features, Layout>(options);
            });
        if (kernel_ == nullptr) throw std::invalid_argument("this ABDADA variant is not compiled in");
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_, options.block_size, 0));
        if (blocks_per_sm == 0) throw std::runtime_error("the ABDADA kernel does not fit on an SM with these settings");
        resident_blocks = blocks_per_sm * ctx.info().sm_count;
        CUDA_CHECK(cudaMalloc(&d_task_, sizeof(SearchTask<Board>)));
    }
    ~AImplT() override { cudaFree(d_task_); }

    void launch(const Position& root, const GameHistory& history, int depth, int alpha, int beta, int blocks,
                const gpu::AbdadaDeviceArgs& a) override {
        // As the CPU searchers' root (host/search_cpu.cpp).
        h_task_.board = to_board<Board>(root);
        h_task_.alpha = short(alpha);
        h_task_.beta = short(beta);
        h_task_.depth = static_cast<signed char>(depth);
        h_task_.ply = 0;
        h_task_.flags = u8(kTaskOnPv | (history.root_repeated ? kTaskRootRepeated : 0));
        h_task_.history = history.root_parent();
        CUDA_CHECK(cudaMemcpyAsync(d_task_, &h_task_, sizeof h_task_, cudaMemcpyHostToDevice, work));
        kernel_<<<blocks, options.block_size, 0, work>>>(d_task_, a);
        CUDA_CHECK(cudaGetLastError());
    }

  private:
    gpu::AbdadaKernel<Board> kernel_ = nullptr;
    SearchTask<Board>* d_task_ = nullptr;
    SearchTask<Board> h_task_{};  // the upload's source: lives until the copy is done
};

template <class Board>
std::unique_ptr<GpuAbdadaDevice::Impl> make_a_impl_for_board(const DeviceContext& ctx, const GpuSearchOptions& o,
                                                            int busy_bits) {
#if CHESSGPU_WITH_MATERIAL_EVAL
    if (o.eval == EvalKind::Material) return std::make_unique<AImplT<Board, MaterialEval>>(ctx, o, busy_bits);
#endif
#if CHESSGPU_WITH_PESTO_EVAL
    if (o.eval == EvalKind::Pesto) return std::make_unique<AImplT<Board, PestoEval>>(ctx, o, busy_bits);
#endif
#if CHESSGPU_WITH_TERMS_EVAL
    if (o.eval == EvalKind::Terms) return std::make_unique<AImplT<Board, PestoTermsEval>>(ctx, o, busy_bits);
#endif
#if CHESSGPU_WITH_NNUE_EVAL
    if (o.eval == EvalKind::Nnue) return std::make_unique<AImplT<Board, NnueEval>>(ctx, o, busy_bits);
#endif
    throw std::invalid_argument("eval not compiled in");
}

std::unique_ptr<GpuAbdadaDevice::Impl> make_a_impl(const DeviceContext& ctx, const GpuSearchOptions& o, int busy_bits) {
    validate_shared(o);
#if CHESSGPU_WITH_STD_BOARD
    if (o.board == BoardKind::Std) return make_a_impl_for_board<StdBoard>(ctx, o, busy_bits);
#endif
#if CHESSGPU_WITH_QUAD_BOARD
    if (o.board == BoardKind::Quad) return make_a_impl_for_board<QuadBoard>(ctx, o, busy_bits);
#endif
    throw std::invalid_argument("board not compiled in");
}

}  // namespace

GpuAbdadaDevice::GpuAbdadaDevice(const DeviceContext& ctx, const GpuSearchOptions& options, int busy_bits)
    : impl_(make_a_impl(ctx, options, busy_bits)) {}

GpuAbdadaDevice::~GpuAbdadaDevice() = default;

FixedDepthResult GpuAbdadaDevice::search(const Position& root, const GameHistory& history, int depth,
                                         const AbdadaDeviceOptions& abdada, const std::vector<Move>& pv_hint,
                                         AbdadaDeviceStats* stats, int alpha, int beta) {
    return impl_->search(root, history, depth, abdada, pv_hint, stats, alpha, beta);
}

unsigned GpuAbdadaDevice::max_workers(int lanes) const { return impl_->max_workers(lanes); }
size_t GpuAbdadaDevice::claims() const { return impl_->claims(); }
void GpuAbdadaDevice::request_stop() { impl_->stop_requested.store(true); }
void GpuAbdadaDevice::clear_stop() { impl_->stop_requested.store(false); }
void GpuAbdadaDevice::set_deadline(std::chrono::steady_clock::time_point deadline) { impl_->deadline = deadline; }
void GpuAbdadaDevice::set_tt(const TTDesc& tt) { impl_->tt = tt; }
const GpuSearchOptions& GpuAbdadaDevice::options() const { return impl_->options; }
std::string GpuAbdadaDevice::describe() const { return "gpu " + cb::describe(impl_->options); }

}  // namespace cb
