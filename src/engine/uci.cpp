#include "engine/uci.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

#include "host/fen.h"
#include "host/host_tables.h"
#include "host/uci_move.h"

namespace cb {

namespace {

std::vector<std::string> split_words(const std::string& line) {
    std::vector<std::string> words;
    std::istringstream in(line);
    for (std::string w; in >> w;) words.push_back(w);
    return words;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// Parses a non-negative integer; returns false if `text` isn't one.
bool parse_count(const std::string& text, long long& out) {
    if (text.empty() || text.size() > 18) return false;
    long long v = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    out = v;
    return true;
}

// Like parse_count, but allows a leading minus (clock times can go negative).
bool parse_signed(const std::string& text, long long& out) {
    if (!text.empty() && text[0] == '-') {
        if (!parse_count(text.substr(1), out)) return false;
        out = -out;
        return true;
    }
    return parse_count(text, out);
}

std::string pv_text(const std::vector<Move>& pv) {
    std::string s;
    for (const Move m : pv) s += " " + move_to_uci(m);
    return s;
}

// `multipv` > 0: the line's rank (MultiPV).
std::string info_line(int depth, int score, u64 nodes, double ms, const std::vector<Move>& pv, int multipv = 0) {
    return "info depth " + std::to_string(depth) + (multipv > 0 ? " multipv " + std::to_string(multipv) : "") +
           " score " + uci_score(score) + " nodes " + std::to_string(nodes) + " nps " +
           std::to_string(u64(double(nodes) * 1000.0 / std::max(ms, 1.0))) + " time " + std::to_string(u64(ms)) +
           " pv" + pv_text(pv);
}

}  // namespace

std::string uci_score(int score) {
    if (score > kScoreMateBound) return "mate " + std::to_string((kScoreMate - score + 1) / 2);
    if (score < -kScoreMateBound) return "mate " + std::to_string(-((kScoreMate + score) / 2));  // "mate 0": mated now
    return "cp " + std::to_string(score);
}

UciEngine::UciEngine(std::ostream& out) : out_(out) {
    std::string error;
    parse_fen(kStartFen, root_, error);
}

UciEngine::~UciEngine() {
    stop();
    finish_search();
}

void UciEngine::send(const std::string& line) {
    const std::lock_guard<std::mutex> lock(out_mutex_);
    out_ << line << '\n';
    out_.flush();
}

bool UciEngine::handle(const std::string& line) {
    const std::vector<std::string> words = split_words(line);
    if (words.empty()) return true;
    const std::string& cmd = words[0];
    try {
        if (cmd == "uci") uci();
        else if (cmd == "isready") is_ready();
        else if (cmd == "ucinewgame") {
            finish_search();
            std::string error;
            parse_fen(kStartFen, root_, error);
            history_ = GameHistory{};
            if (backend_ && !backend_stale_) backend_->clear_hash();
        } else if (cmd == "setoption") set_option(words);
        else if (cmd == "position") position(words);
        else if (cmd == "go") go(words);
        else if (cmd == "stop") stop();
        else if (cmd == "ponderhit") {
            // Pondering isn't offered (no Ponder option), so nothing to do.
        } else if (cmd == "quit") {
            stop();
            finish_search();
            return false;
        } else if (cmd == "d") {
            send("info string fen " + format_fen(root_));
        } else if (cmd == "legalmoves") {
            // Debugging and tools (scripts/make_positions.py): the legal moves of the current position.
            const HqAttacks a(host_attack_tables());
            MoveList list;
            generate_legal(to_board<QuadBoard>(root_), a, list);
            std::string text = "info string legalmoves";
            for (int i = 0; i < list.size; ++i) text += " " + move_to_uci(list.moves[i]);
            send(text);
        } else {
            send("info string unknown command: " + cmd);
        }
    } catch (const std::exception& e) {
        send(std::string("info string error: ") + e.what());
    }
    return true;
}

void UciEngine::uci() {
    send("id name ChessGPU");
    send("id author the ChessGPU authors");
    send("option name Search type combo default " + std::string(name_of(options_.search)) +
         " var cpu var gpu-independent var ybwc-host var ybwc-device var abdada-cpu var abdada-device");
    send("option name Threads type spin default " + std::to_string(options_.threads) + " min 1 max 256");
    send("option name Workers type spin default " + std::to_string(options_.workers) + " min 1 max 1048576");
    send("option name AbdadaMinDepth type spin default " + std::to_string(options_.abdada.min_depth) +
         " min 1 max 20");
    send(std::string("option name AbdadaDeferFirst type check default ") +
         (options_.abdada.defer_first ? "true" : "false"));
    send("option name Board type combo default quad var quad var std");
    send("option name Model type combo default thread var thread var warp");
    send(std::string("option name Eval type combo default ") + name_of(options_.gpu.eval) +
         " var pesto var material var terms var nnue");
    send("option name Ordering type combo default staged var staged var list");
    send("option name Stack type combo default local var local var shared");
    std::string features = std::string("option name Features type combo default ") + name_of(options_.gpu.features);
    for (const FeatureKind f : kAllFeatureKinds) features += std::string(" var ") + name_of(f);
    send(features);
    send(std::string("option name Attacks type combo default ") + name_of(options_.gpu.attacks) +
         " var hq var magic");
    send(std::string("option name TTLayout type combo default ") + name_of(options_.gpu.tt_layout) +
         " var packed64 var wide128");
    send("option name Hash type spin default " + std::to_string(options_.hash_mb) + " min 1 max " +
         std::to_string(kMaxHashMb));
    send("option name Clear Hash type button");
    send("option name NullMinDepth type spin default " + std::to_string(options_.gpu.prune.null_min_depth) +
         " min 1 max " + std::to_string(kMaxNullMinDepth));
    send("option name NullReduction type spin default " + std::to_string(options_.gpu.prune.null_base) +
         " min 1 max " + std::to_string(kMaxNullBase));
    send("option name NullDivisor type spin default " + std::to_string(options_.gpu.prune.null_divisor) +
         " min 0 max " + std::to_string(kMaxNullDivisor));
    send("option name LmrMinDepth type spin default " + std::to_string(options_.gpu.prune.lmr_min_depth) +
         " min 1 max " + std::to_string(kMaxLmrMinDepth));
    send("option name LmrMoves type spin default " + std::to_string(options_.gpu.prune.lmr_moves) + " min 0 max " +
         std::to_string(kMaxLmrMoves));
    send("option name LmrBase type spin default " + std::to_string(options_.gpu.prune.lmr_base) + " min 0 max 255");
    send("option name LmrDivisor type spin default " + std::to_string(options_.gpu.prune.lmr_divisor) +
         " min 1 max 255");
    send("option name FutDepth type spin default " + std::to_string(options_.gpu.prune.fut_depth) + " min 1 max " +
         std::to_string(kMaxFutDepth));
    send("option name FutMargin type spin default " + std::to_string(options_.gpu.prune.fut_margin) +
         " min 0 max 255");
    send("option name DeltaMargin type spin default " + std::to_string(options_.gpu.prune.delta_margin) +
         " min 0 max " + std::to_string(kMaxDeltaMargin));
    send(std::string("option name SeePrune type check default ") + (options_.gpu.prune.see_prune ? "true" : "false"));
    send("option name SplitDepth type spin default " + std::to_string(options_.split_depth) + " min 0 max 8");
    send("option name LeafDepth type spin default " + std::to_string(options_.ybwc.leaf_depth) + " min 0 max 20");
    send("option name ExplicitPlies type spin default " + std::to_string(options_.ybwc.max_explicit_ply) +
         " min 0 max 32");
    send(std::string("option name CutWaves type check default ") + (options_.ybwc.cut_waves ? "true" : "false"));
    send("option name NodeBudget type spin default " + std::to_string(options_.ybwc.node_budget) +
         " min 0 max 100000000");
    send("option name Lanes type spin default " + std::to_string(options_.device_limits.lanes) + " min 1 max 32");
    send("option name BlockSize type spin default " + std::to_string(options_.gpu.block_size) + " min 32 max 1024");
    send("option name MoveOverhead type spin default 50 min 0 max 5000");
    send("option name Aspiration type spin default " + std::to_string(options_.aspiration_cp) + " min 0 max 1000");
    send("option name FailLowTime type spin default " + std::to_string(options_.fail_low_percent) + " min 0 max 400");
    send("option name MultiPV type spin default 1 min 1 max " + std::to_string(kMaxMultiPv));
    send("uciok");
}

void UciEngine::is_ready() {
    ensure_backend();
    send("readyok");
}

void UciEngine::set_option(const std::vector<std::string>& w) {
    // setoption name <name words> [value <value words>]
    std::string name;
    std::string value;
    size_t i = 1;
    if (i < w.size() && lower(w[i]) == "name") ++i;
    for (; i < w.size() && lower(w[i]) != "value"; ++i) name += (name.empty() ? "" : " ") + w[i];
    if (i < w.size()) ++i;
    for (; i < w.size(); ++i) value += (value.empty() ? "" : " ") + w[i];
    const std::string key = lower(name);
    const std::string v = lower(value);
    {
        const std::lock_guard<std::mutex> lock(search_mutex_);
        if (searching_) {
            send("info string options can't change during a search");
            return;
        }
    }
    bool ok = true;
    long long n = 0;
    if (key == "search") ok = parse_search_kind(v, options_.search);
    else if (key == "threads") {
        ok = parse_count(v, n) && n >= 1 && n <= 256;
        if (ok) options_.threads = int(n);
    } else if (key == "aspiration") {
        ok = parse_count(v, n) && n <= 1000;
        if (ok) options_.aspiration_cp = int(n);
    } else if (key == "faillowtime") {
        ok = parse_count(v, n) && n <= 400;
        if (ok) options_.fail_low_percent = int(n);
    } else if (key == "workers") {
        ok = parse_count(v, n) && n >= 1 && n <= 1048576;
        if (ok) options_.workers = unsigned(n);
    } else if (key == "abdadamindepth") {
        ok = parse_count(v, n) && n >= 1 && n <= 20;
        if (ok) options_.abdada.min_depth = u8(n);
    } else if (key == "abdadadeferfirst") {
        ok = v == "true" || v == "false";
        if (ok) options_.abdada.defer_first = v == "true" ? 1 : 0;
    }
    else if (key == "board") ok = parse_board_kind(v, options_.gpu.board);
    else if (key == "model") ok = parse_exec_model(v, options_.gpu.model);
    else if (key == "eval") ok = parse_eval_kind(v, options_.gpu.eval);
    else if (key == "attacks") ok = (v == "hq" || v == "magic") && parse_attacks_kind(v, options_.gpu.attacks);
    else if (key == "ordering") {
        ok = parse_ordering_kind(v, options_.gpu.ordering);
        if (ok) options_.cpu_ordering = options_.gpu.ordering;
    } else if (key == "stack") ok = parse_stack_kind(v, options_.gpu.stack);
    else if (key == "features") ok = parse_feature_kind(v, options_.gpu.features);
    else if (key == "ttlayout") ok = parse_tt_layout_kind(v, options_.gpu.tt_layout);
    else if (key == "hash") {
        ok = parse_count(v, n) && n >= 1 && n <= (long long)kMaxHashMb;
        if (ok) options_.hash_mb = size_t(n);
    } else if (key == "nullmindepth") {
        ok = parse_count(v, n) && n >= 1 && n <= kMaxNullMinDepth;
        if (ok) options_.gpu.prune.null_min_depth = u8(n);
    } else if (key == "nullreduction") {
        ok = parse_count(v, n) && n >= 1 && n <= kMaxNullBase;
        if (ok) options_.gpu.prune.null_base = u8(n);
    } else if (key == "nulldivisor") {
        ok = parse_count(v, n) && n <= kMaxNullDivisor;
        if (ok) options_.gpu.prune.null_divisor = u8(n);
    } else if (key == "lmrmindepth") {
        ok = parse_count(v, n) && n >= 1 && n <= kMaxLmrMinDepth;
        if (ok) options_.gpu.prune.lmr_min_depth = u8(n);
    } else if (key == "lmrmoves") {
        ok = parse_count(v, n) && n <= kMaxLmrMoves;
        if (ok) options_.gpu.prune.lmr_moves = u8(n);
    } else if (key == "lmrbase") {
        ok = parse_count(v, n) && n <= 255;
        if (ok) options_.gpu.prune.lmr_base = u8(n);
    } else if (key == "lmrdivisor") {
        ok = parse_count(v, n) && n >= 1 && n <= 255;
        if (ok) options_.gpu.prune.lmr_divisor = u8(n);
    } else if (key == "futdepth") {
        ok = parse_count(v, n) && n >= 1 && n <= kMaxFutDepth;
        if (ok) options_.gpu.prune.fut_depth = u8(n);
    } else if (key == "futmargin") {
        ok = parse_count(v, n) && n <= 255;
        if (ok) options_.gpu.prune.fut_margin = u8(n);
    } else if (key == "deltamargin") {
        ok = parse_count(v, n) && n <= kMaxDeltaMargin;
        if (ok) options_.gpu.prune.delta_margin = u16(n);
    } else if (key == "seeprune") {
        ok = v == "true" || v == "false";
        if (ok) options_.gpu.prune.see_prune = v == "true" ? 1 : 0;
    } else if (key == "clear hash") {
        if (backend_ && !backend_stale_) backend_->clear_hash();
        return;  // a button: nothing to rebuild
    } else if (key == "multipv") {
        if (!parse_count(v, n) || n < 1 || n > kMaxMultiPv) {
            send("info string invalid value for " + name + ": " + value);
            return;
        }
        multipv_ = int(n);
        return;  // a search setting: nothing to rebuild
    }
    else if (key == "splitdepth") {
        ok = parse_count(v, n) && n <= 8;
        if (ok) options_.split_depth = int(n);
    } else if (key == "leafdepth") {
        ok = parse_count(v, n) && n <= 20;
        if (ok) options_.ybwc.leaf_depth = int(n);
    } else if (key == "explicitplies") {
        ok = parse_count(v, n) && n <= 32;
        if (ok) options_.ybwc.max_explicit_ply = int(n);
    } else if (key == "lanes") {
        ok = parse_count(v, n) && n >= 1 && n <= 32;
        if (ok) options_.device_limits.lanes = int(n);
    } else if (key == "nodebudget") {
        ok = parse_count(v, n) && n <= 100000000;
        if (ok) options_.ybwc.node_budget = u32(n);
    } else if (key == "cutwaves") {
        ok = v == "true" || v == "false";
        if (ok) options_.ybwc.cut_waves = v == "true";
    } else if (key == "blocksize") {
        ok = parse_count(v, n) && n >= 32 && n <= 1024 && n % 32 == 0;
        if (ok) options_.gpu.block_size = int(n);
    } else if (key == "moveoverhead") {
        ok = parse_count(v, n) && n <= 5000;
        if (ok) move_overhead_ms_ = double(n);
    } else {
        send("info string unknown option: " + name);
        return;
    }
    if (!ok) {
        send("info string invalid value for " + name + ": " + value);
        return;
    }
    backend_stale_ = true;
}

void UciEngine::position(const std::vector<std::string>& w) {
    finish_search();
    // position startpos [moves ...] | position fen <fields> [moves ...]
    size_t i = 1;
    std::string fen;
    if (i < w.size() && w[i] == "startpos") {
        fen = kStartFen;
        ++i;
    } else if (i < w.size() && w[i] == "fen") {
        for (++i; i < w.size() && w[i] != "moves"; ++i) fen += (fen.empty() ? "" : " ") + w[i];
    } else {
        throw std::invalid_argument("position needs startpos or fen");
    }
    std::string moves;
    if (i < w.size() && w[i] == "moves")
        for (++i; i < w.size(); ++i) moves += (moves.empty() ? "" : " ") + w[i];
    Position start;
    std::string error;
    if (!parse_fen(fen, start, error)) throw std::invalid_argument("invalid FEN: " + error);
    Position root;
    GameHistory history;
    if (!build_game(start, moves, root, history, error)) throw std::invalid_argument(error);
    root_ = root;
    history_ = std::move(history);
}

void UciEngine::ensure_backend() {
    {
        const std::lock_guard<std::mutex> lock(search_mutex_);
        if (searching_) return;  // never swap the backend under a running search
    }
    if (backend_ && !backend_stale_) return;
    backend_.reset();
    if (needs_gpu(options_.search) && !ctx_) ctx_ = std::make_unique<DeviceContext>();
    backend_ = make_backend(options_, ctx_.get());
    backend_stale_ = false;
}

void UciEngine::go(const std::vector<std::string>& w) {
    finish_search();
    TimeControl tc;
    tc.overhead_ms = move_overhead_ms_;
    SearchRequest request;
    request.root = root_;
    request.history = history_;
    bool infinite = false;
    const Color us = state::side(root_.state);
    for (size_t i = 1; i < w.size(); ++i) {
        long long n = 0;
        const bool has_value = i + 1 < w.size();
        const std::string& key = w[i];
        if (key == "infinite") {
            infinite = true;
        } else if (key == "ponder") {
            // not offered; searched like a normal go
        } else if (key == "searchmoves") {
            break;  // not supported: the rest of the line lists moves
        } else if (has_value && parse_signed(w[i + 1], n)) {
            ++i;
            if (key == "depth") request.max_depth = int(std::clamp<long long>(n, 1, kMaxPly - 1));
            else if (key == "nodes") request.max_nodes = u64(std::max<long long>(n, 1));
            else if (key == "movetime") tc.move_time_ms = double(std::max<long long>(n, 0));
            else if (key == "wtime" && us == WHITE) tc.time_ms = double(n);
            else if (key == "btime" && us == BLACK) tc.time_ms = double(n);
            else if (key == "winc" && us == WHITE) tc.inc_ms = double(n);
            else if (key == "binc" && us == BLACK) tc.inc_ms = double(n);
            else if (key == "movestogo") tc.moves_to_go = int(n);
            // the opponent's clock and "mate" are ignored
        }
    }
    if (tc.time_ms < 0 && (w.end() != std::find(w.begin(), w.end(), us == WHITE ? "wtime" : "btime")))
        tc.time_ms = 0;  // a negative clock: move at once
    const TimeBudget budget = plan_time(tc, options_.fail_low_percent);
    const Clock::time_point now = Clock::now();
    const auto at = [&](double ms) {
        return ms < 0 ? Clock::time_point::max()
                      : now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
    };
    request.soft = at(budget.soft_ms);
    request.hard = at(budget.hard_ms);
    if (budget.fail_low_ms > budget.hard_ms) request.fail_low = at(budget.fail_low_ms);
    request.aspiration_cp = options_.aspiration_cp;
    request.multipv = multipv_;

    try {
        ensure_backend();
    } catch (const std::exception& e) {
        // Keep playing: fall back to the CPU search for this move.
        send(std::string("info string backend error: ") + e.what() + "; searching on the CPU");
        EngineOptions cpu = options_;
        cpu.search = SearchKind::Cpu;
        backend_ = make_backend(cpu, nullptr);
        backend_stale_ = true;
    }
    backend_->clear_stop();
    {
        const std::lock_guard<std::mutex> lock(search_mutex_);
        searching_ = true;
        stop_received_ = false;
    }
    SearchBackend* backend = backend_.get();
    search_thread_ = std::thread([this, backend, request, infinite] {
        send("info string search " + backend->describe());
        u64 reported_nodes = 0;
        const SearchOutcome outcome = iterative_deepening(*backend, request, [&](const IterationReport& r) {
            reported_nodes = r.total_nodes;
            if (r.lines.empty()) send(info_line(r.depth, r.score, r.total_nodes, r.ms, r.pv));
            for (size_t i = 0; i < r.lines.size(); ++i)
                send(info_line(r.depth, r.lines[i].score, r.total_nodes, r.ms, r.lines[i].pv, int(i) + 1));
        });
        // An iteration cut short by a limit or "stop" isn't reported, but it
        // took nodes and time. Report the totals, with the deepest finished
        // iteration's score and PV, so GUIs and match runners see what the
        // move really cost (fastchess counts the gap as engine latency).
        if (outcome.depth > 0 && outcome.nodes != reported_nodes)
            send(info_line(outcome.depth, outcome.score, outcome.nodes, outcome.ms, outcome.pv,
                           request.multipv > 1 ? 1 : 0));
        std::unique_lock<std::mutex> lock(search_mutex_);
        // "go infinite" answers only after "stop", even if the search ended.
        if (infinite) search_cv_.wait(lock, [&] { return stop_received_; });
        lock.unlock();
        send("bestmove " + (outcome.best.is_none() ? std::string("(none)") : move_to_uci(outcome.best)));
        lock.lock();
        searching_ = false;
        lock.unlock();
        search_cv_.notify_all();
    });
}

void UciEngine::stop() {
    {
        const std::lock_guard<std::mutex> lock(search_mutex_);
        stop_received_ = true;
        if (searching_ && backend_) backend_->request_stop();
    }
    search_cv_.notify_all();
}

void UciEngine::wait_for_search() {
    std::unique_lock<std::mutex> lock(search_mutex_);
    search_cv_.wait(lock, [&] { return !searching_; });
}

// Stops a running search (a GUI shouldn't send position/go/quit during one,
// but if it does, the search ends and sends its bestmove first) and joins
// the search thread.
void UciEngine::finish_search() {
    bool running;
    {
        const std::lock_guard<std::mutex> lock(search_mutex_);
        running = searching_;
    }
    if (running) stop();
    if (search_thread_.joinable()) search_thread_.join();
}

}  // namespace cb
