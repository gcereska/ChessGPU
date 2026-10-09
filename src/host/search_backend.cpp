#include "host/search_backend.h"

#include <algorithm>

#include "host/multipv.h"

namespace cb {

SearchOutcome iterative_deepening(SearchBackend& backend, const SearchRequest& request,
                                  const std::function<void(const IterationReport&)>& report) {
    if (request.multipv > 1) return multipv_deepening(backend, request, report);
    const Clock::time_point start = Clock::now();
    const auto elapsed_ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); };
    SearchOutcome out;
    std::vector<Move> pv;
    const int max_depth = std::clamp(request.max_depth, 1, kMaxPly - 1);
    backend.new_search();
    backend.set_deadline(Clock::time_point::max());
    for (int depth = 1; depth <= max_depth; ++depth) {
        if (depth > 1) {
            if (Clock::now() >= request.soft) break;
            if (request.max_nodes && out.nodes >= request.max_nodes) break;
            backend.set_deadline(request.hard);
        }
        FixedDepthResult r;
        const bool aspirate = request.aspiration_cp > 0 && depth >= kAspirationMinDepth && backend.honors_window() &&
                              !out.best.is_none() && !is_mate_score(out.score);
        if (aspirate) {
            // Aspiration: the previous score +- the margin; a fail widens
            // that side around the fail-soft bound, the margin doubling.
            int delta = request.aspiration_cp;
            int alpha = std::max(-kScoreInfinite, out.score - delta);
            int beta = std::min(kScoreInfinite, out.score + delta);
            for (;;) {
                r = backend.search_window(request.root, request.history, depth, pv, alpha, beta);
                out.nodes += r.search.nodes;
                if (r.search.aborted) break;
                const int s = r.search.score;
                if ((s > alpha || alpha == -kScoreInfinite) && (s < beta || beta == kScoreInfinite)) break;
                ++out.researches;
                delta *= 2;
                if (s <= alpha) {
                    alpha = is_mate_score(s) ? -kScoreInfinite : std::max(-kScoreInfinite, s - delta);
                    // The best move so far may be bad: take longer to find a better one.
                    if (request.fail_low > request.hard) backend.set_deadline(request.fail_low);
                } else {
                    beta = is_mate_score(s) ? kScoreInfinite : std::min(kScoreInfinite, s + delta);
                }
            }
        } else {
            r = backend.search_depth(request.root, request.history, depth, pv);
            out.nodes += r.search.nodes;
        }
        if (r.search.aborted) {
            if (depth > 1) break;
            // Stopped during depth 1 (a stop arriving right after "go"):
            // finish depth 1 anyway so there is a move, then stop.
            backend.clear_stop();
            r = backend.search_depth(request.root, request.history, 1, pv);
            out.nodes += r.search.nodes;
            backend.request_stop();
        }
        pv = r.pv;
        out.best = r.search.best;
        out.score = r.search.score;
        out.depth = depth;
        out.pv = r.pv;
        if (report) {
            IterationReport it;
            it.depth = depth;
            it.score = r.search.score;
            it.nodes = r.search.nodes;
            it.total_nodes = out.nodes;
            it.ms = elapsed_ms();
            it.pv = r.pv;
            report(it);
        }
        if (out.best.is_none()) break;  // no legal moves: nothing deeper to find
    }
    backend.set_deadline(Clock::time_point::max());
    out.ms = elapsed_ms();
    return out;
}

CpuBackend::CpuBackend(const CpuSearchOptions& options, size_t hash_mb) : options_(options) {
    if (uses_tt(options.features)) tt_ = std::make_unique<HostTT>(hash_mb);
}

FixedDepthResult CpuBackend::search_window(const Position& root, const GameHistory& history, int depth,
                                           const std::vector<Move>& pv_hint, int alpha, int beta) {
    CpuSearchOptions o = options_;
    if (tt_) o.tt = tt_->desc();
    return cpu_search(root, history, depth, alpha, beta, o, pv_hint, &stop_, deadline_);
}

std::string CpuBackend::describe() const {
    std::string s = "cpu " + cb::describe(options_);
    if (tt_) s += " hash " + std::to_string(tt_->size_mb()) + " MB";
    return s;
}

CpuAbdadaBackend::CpuAbdadaBackend(const CpuSearchOptions& options, int workers, size_t hash_mb,
                                   const AbdadaParams& params)
    : options_(options), workers_(workers), params_(params) {
    if (workers < 1 || workers > 256) throw std::invalid_argument("ABDADA workers must be 1 to 256");
    validate_abdada(params);
    options_.ordering = OrderingKind::List;
    if (uses_tt(options.features)) tt_ = std::make_unique<HostTT>(hash_mb);
}

FixedDepthResult CpuAbdadaBackend::search_window(const Position& root, const GameHistory& history, int depth,
                                                 const std::vector<Move>& pv_hint, int alpha, int beta) {
    CpuSearchOptions o = options_;
    if (tt_) o.tt = tt_->desc();
    return cpu_search_abdada(root, history, depth, alpha, beta, o, pv_hint, workers_, table_, &stop_, deadline_,
                             &stats_, params_);
}

std::string CpuAbdadaBackend::describe() const {
    std::string s = "abdada-cpu " + std::to_string(workers_) + " workers " + cb::describe(options_);
    if (tt_) s += " hash " + std::to_string(tt_->size_mb()) + " MB";
    return s + cb::describe(params_);
}

std::string CpuAbdadaBackend::last_stats() const {
    return "workers " + std::to_string(stats_.workers) + ", winner " + std::to_string(stats_.winner) + ", deferred " +
           std::to_string(stats_.deferred);
}

TimeBudget plan_time(const TimeControl& tc, int fail_low_percent) {
    TimeBudget b;
    if (tc.move_time_ms >= 0) {
        b.hard_ms = std::max(1.0, tc.move_time_ms - tc.overhead_ms);
        b.soft_ms = b.fail_low_ms = b.hard_ms;
        return b;
    }
    if (tc.time_ms < 0) return b;  // no clock: depth, nodes or infinite
    if (tc.time_ms <= tc.overhead_ms) {
        b.soft_ms = b.hard_ms = b.fail_low_ms = 0;  // out of time: depth 1 only (it always finishes)
        return b;
    }
    const double usable = tc.time_ms - tc.overhead_ms;
    // Spread the clock over the moves left (30 when unknown), plus most of
    // the increment.
    const int moves_left = tc.moves_to_go > 0 ? tc.moves_to_go : 30;
    const double optimum = std::min(usable / moves_left + tc.inc_ms * 0.8, usable * 0.8);
    // An iteration usually takes several times longer than the one before,
    // so a new one starts only while well under the optimum, and a running
    // one may go a little over it.
    b.hard_ms = std::max(1.0, std::min(optimum * 3.0, usable * 0.8));
    b.soft_ms = std::max(1.0, std::min(optimum * 0.6, b.hard_ms));
    b.fail_low_ms = std::max(b.hard_ms, std::min(b.hard_ms * (1.0 + fail_low_percent / 100.0), usable * 0.4));
    return b;
}

}  // namespace cb
