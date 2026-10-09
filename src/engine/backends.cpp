#include "engine/backends.h"

#include <algorithm>
#include <stdexcept>

#include "host/split_search.h"

namespace cb {

const char* name_of(SearchKind k) {
    switch (k) {
        case SearchKind::Cpu: return "cpu";
        case SearchKind::GpuIndependent: return "gpu-independent";
        case SearchKind::YbwcHost: return "ybwc-host";
        case SearchKind::YbwcDevice: return "ybwc-device";
        case SearchKind::AbdadaCpu: return "abdada-cpu";
        case SearchKind::AbdadaDevice: return "abdada-device";
    }
    return "?";
}

bool parse_search_kind(std::string_view text, SearchKind& out) {
    if (text == "cpu") out = SearchKind::Cpu;
    else if (text == "gpu-independent") out = SearchKind::GpuIndependent;
    else if (text == "ybwc-host") out = SearchKind::YbwcHost;
    else if (text == "ybwc-device") out = SearchKind::YbwcDevice;
    else if (text == "abdada-cpu") out = SearchKind::AbdadaCpu;
    else if (text == "abdada-device") out = SearchKind::AbdadaDevice;
    else return false;
    return true;
}

namespace {

// The table for feature sets that use one (device memory), else null.
std::unique_ptr<DeviceTT> make_device_tt(const EngineOptions& o) {
    return uses_tt(o.gpu.features) ? std::make_unique<DeviceTT>(o.hash_mb) : nullptr;
}

std::string hash_text(const std::unique_ptr<DeviceTT>& tt) {
    return tt ? " hash " + std::to_string(tt->size_mb()) + " MB" : "";
}

// A parallel search over a task runner (the GPU searcher).
class RunnerBackend final : public SearchBackend {
  public:
    RunnerBackend(std::unique_ptr<TaskRunner> runner, const EngineOptions& options)
        : runner_(std::move(runner)), options_(options), tt_(make_device_tt(options)) {}

    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        if (tt_) runner_->set_tt(tt_->desc());
        if (options_.search == SearchKind::GpuIndependent) {
            SplitSearchStats s;
            FixedDepthResult r = search_independent(*runner_, root, history, depth, options_.split_depth, pv_hint, &s);
            stats_ = "explicit " + std::to_string(s.explicit_nodes) + " | tasks " + std::to_string(s.tasks);
            return r;
        }
        YbwcStats s;
        FixedDepthResult r = search_ybwc(*runner_, root, history, depth, options_.ybwc, pv_hint, &s);
        stats_ = cb::describe(s);
        return r;
    }
    std::string last_stats() const override { return stats_; }
    void request_stop() override { runner_->request_stop(); }
    void clear_stop() override { runner_->clear_stop(); }
    void set_deadline(Clock::time_point deadline) override { runner_->set_deadline(deadline); }
    std::string describe() const override {
        std::string s = std::string(name_of(options_.search)) + " " + runner_->describe() + hash_text(tt_);
        if (options_.search == SearchKind::GpuIndependent) return s + " split " + std::to_string(options_.split_depth);
        return s + " leaf " + std::to_string(options_.ybwc.leaf_depth) + " plies " +
               std::to_string(options_.ybwc.max_explicit_ply);
    }
    void new_search() override {
        if (tt_) tt_->new_search();
    }
    void clear_hash() override {
        if (tt_) tt_->clear();
    }

  private:
    std::unique_ptr<TaskRunner> runner_;
    EngineOptions options_;
    std::unique_ptr<DeviceTT> tt_;
    std::string stats_;
};

// YBWC run entirely on the GPU.
class DeviceYbwcBackend final : public SearchBackend {
  public:
    DeviceYbwcBackend(const DeviceContext& ctx, const EngineOptions& options)
        : search_(ctx, options.gpu, options.device_limits), options_(options), tt_(make_device_tt(options)) {}

    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        return search_window(root, history, depth, pv_hint, -kScoreInfinite, kScoreInfinite);
    }
    bool honors_window() const override { return true; }
    FixedDepthResult search_window(const Position& root, const GameHistory& history, int depth,
                                   const std::vector<Move>& pv_hint, int alpha, int beta) override {
        if (tt_) search_.set_tt(tt_->desc());
        // A tree too big for the pools: split more coarsely (bigger tasks,
        // fewer explicit nodes) and search the iteration again.
        YbwcOptions o = options_.ybwc;
        YbwcDeviceStats s;
        FixedDepthResult r = search_.search(root, history, depth, o, pv_hint, &s, alpha, beta);
        stats_ = cb::describe(s);
        for (int retry = 1; retry <= 3 && r.search.aborted && s.overflow; ++retry) {
            const u64 searched = r.search.nodes;
            o.node_budget = o.node_budget ? o.node_budget * 4 : 0;
            o.leaf_depth += 1;
            r = search_.search(root, history, depth, o, pv_hint, &s, alpha, beta);
            r.search.nodes += searched;
            stats_ += " | pools full, retry " + std::to_string(retry) + " with leaf " + std::to_string(o.leaf_depth) +
                      " budget " + std::to_string(o.node_budget) + ": " + cb::describe(s);
        }
        return r;
    }
    void request_stop() override { search_.request_stop(); }
    void clear_stop() override { search_.clear_stop(); }
    void set_deadline(Clock::time_point deadline) override { search_.set_deadline(deadline); }
    std::string describe() const override {
        return "ybwc-device " + search_.describe() + hash_text(tt_) + " leaf " +
               std::to_string(options_.ybwc.leaf_depth) + " plies " + std::to_string(options_.ybwc.max_explicit_ply) +
               " budget " + std::to_string(options_.ybwc.node_budget) + (options_.ybwc.cut_waves ? " waves" : "") +
               " lanes " + std::to_string(options_.device_limits.lanes);
    }
    std::string last_stats() const override { return stats_; }
    void new_search() override {
        if (tt_) tt_->new_search();
    }
    void clear_hash() override {
        if (tt_) tt_->clear();
    }

  private:
    GpuYbwcDevice search_;
    EngineOptions options_;
    std::unique_ptr<DeviceTT> tt_;
    std::string stats_;
};

// ABDADA on the GPU (gpu/abdada_device.h): list ordering whatever the options say.
class DeviceAbdadaBackend final : public SearchBackend {
  public:
    DeviceAbdadaBackend(const DeviceContext& ctx, const EngineOptions& options)
        : search_(ctx, list_ordering(options.gpu)), options_(options), tt_(make_device_tt(options)) {}

    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        return search_window(root, history, depth, pv_hint, -kScoreInfinite, kScoreInfinite);
    }
    bool honors_window() const override { return true; }
    FixedDepthResult search_window(const Position& root, const GameHistory& history, int depth,
                                   const std::vector<Move>& pv_hint, int alpha, int beta) override {
        if (tt_) search_.set_tt(tt_->desc());
        AbdadaDeviceStats s;
        FixedDepthResult r = search_.search(root, history, depth, abdada_options(), pv_hint, &s, alpha, beta);
        stats_ = cb::describe(s);
        return r;
    }
    void request_stop() override { search_.request_stop(); }
    void clear_stop() override { search_.clear_stop(); }
    void set_deadline(Clock::time_point deadline) override { search_.set_deadline(deadline); }
    std::string describe() const override {
        const AbdadaDeviceOptions a = abdada_options();
        return "abdada-device " + search_.describe() + hash_text(tt_) + " workers " +
               std::to_string(std::min(a.workers, search_.max_workers(a.lanes))) + " lanes " + std::to_string(a.lanes) +
               cb::describe(a.params);
    }
    std::string last_stats() const override { return stats_; }
    void new_search() override {
        if (tt_) tt_->new_search();
    }
    void clear_hash() override {
        if (tt_) tt_->clear();
    }

  private:
    static GpuSearchOptions list_ordering(GpuSearchOptions o) {
        o.model = ExecModel::Thread;
        o.ordering = OrderingKind::List;
        o.stack = StackKind::Local;
        return o;
    }
    AbdadaDeviceOptions abdada_options() const {
        return AbdadaDeviceOptions{options_.workers, options_.device_limits.lanes, options_.abdada};
    }

    GpuAbdadaDevice search_;
    EngineOptions options_;
    std::unique_ptr<DeviceTT> tt_;
    std::string stats_;
};

}  // namespace

std::unique_ptr<SearchBackend> make_backend(const EngineOptions& options, const DeviceContext* ctx) {
    if (options.search == SearchKind::Cpu) {
        CpuSearchOptions cpu;
        cpu.board = options.gpu.board;
        cpu.eval = options.gpu.eval;
        cpu.ordering = options.cpu_ordering;
        cpu.features = options.gpu.features;
        cpu.tt_layout = options.gpu.tt_layout;
        cpu.prune = options.gpu.prune;
        return std::make_unique<CpuBackend>(cpu, options.hash_mb);
    }
    if (options.search == SearchKind::AbdadaCpu) {
        CpuSearchOptions cpu;
        cpu.board = options.gpu.board;
        cpu.eval = options.gpu.eval;
        cpu.features = options.gpu.features;
        cpu.tt_layout = options.gpu.tt_layout;
        cpu.prune = options.gpu.prune;
        return std::make_unique<CpuAbdadaBackend>(cpu, options.threads, options.hash_mb, options.abdada);
    }
    if (options.hash_mb < 1 || options.hash_mb > kMaxHashMb)
        throw std::invalid_argument("hash size must be between 1 and " + std::to_string(kMaxHashMb) + " MB");
    if (ctx == nullptr) throw std::invalid_argument("the GPU search needs a device context");
    if (options.split_depth < 0 || options.split_depth > 8)
        throw std::invalid_argument("split depth must be between 0 and 8");
    if (options.ybwc.leaf_depth < 0 || options.ybwc.leaf_depth > 20 || options.ybwc.max_explicit_ply < 0 ||
        options.ybwc.max_explicit_ply > 32)
        throw std::invalid_argument("YBWC leaf depth must be 0-20 and explicit plies 0-32");
    if (options.search == SearchKind::YbwcDevice) return std::make_unique<DeviceYbwcBackend>(*ctx, options);
    if (options.search == SearchKind::AbdadaDevice) {
        if (options.workers < 1) throw std::invalid_argument("ABDADA needs at least one worker");
        return std::make_unique<DeviceAbdadaBackend>(*ctx, options);
    }
    return std::make_unique<RunnerBackend>(std::make_unique<GpuSearcher>(*ctx, options.gpu), options);
}

}  // namespace cb
