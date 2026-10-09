// ABDADA on the GPU (gpu/abdada_device.h; tests/test_abdada.cpp for the CPU
// prototype, whose search_task hooks the workers share):
// - one worker searches exactly like the CPU's sequential search with list
//   ordering: same score, best move and node count, with and without a table,
//   in every board, eval and attack variant, with zero windows, a PV hint and
//   a game history too; it releases every claim, and claims left by an
//   earlier, stopped search don't count (the table starts empty);
// - many workers without a table return the sequential score (every
//   worker's search is exact), with a best move that scores as much, and
//   defer moves, with one worker per warp and with 32; every worker runs,
//   each with its own number, and finishing stops the others;
// - with a shared table, iterative deepening ends every iteration with a
//   legal move, and most root scores equal the sequential search's;
// - stop requests and past deadlines end the search (aborted); asking for
//   more workers than fit gives the most that fit.

#include <string>
#include <vector>

#include "gpu/abdada_device.h"
#include "gpu/device_tt.h"
#include "gpu/perft_gpu.h"
#include "gpu/search_gpu.h"
#include "gpu/variants.h"
#include "gpu_search_util.h"
#include "gpu_test_env.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

constexpr size_t kTableMb = 16;  // same size on both sides: same buckets

GpuSearchOptions abdada_variant(BoardKind board, EvalKind eval, FeatureKind features, AttacksKind attacks) {
    GpuSearchOptions o;
    o.board = board;
    o.eval = eval;
    o.model = ExecModel::Thread;
    o.ordering = OrderingKind::List;
    o.features = features;
    o.tt_layout = TTLayoutKind::Wide128;
    o.attacks = attacks;
    return o;
}

CpuSearchOptions sequential(BoardKind board, EvalKind eval, FeatureKind features, const HostTT* table) {
    CpuSearchOptions o{board, eval, OrderingKind::List};
    o.features = features;
    if (table) o.tt = table->desc();
    return o;
}

bool compiled(const GpuSearchOptions& o) {
#if CHESSGPU_WITH_ABDADA_DEVICE_SEARCH  // its own kernel, whatever CHESSGPU_MODELS and CHESSGPU_ORDERINGS say
    return is_compiled(o.board) && is_compiled(o.eval) && is_compiled(o.features, o.tt_layout) &&
           is_compiled_for_search(o.attacks);
#else
    (void)o;
    return false;
#endif
}

BoardKind first_board() { return is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front(); }

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

bool is_legal_move(const Position& p, Move m) {
    MoveList list;
    generate_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks(), list);
    for (int i = 0; i < list.size; ++i)
        if (list.moves[i] == m) return true;
    return false;
}

std::string result_text(const FixedDepthResult& r) {
    return std::to_string(r.search.score) + " " + move_to_uci(r.search.best) + " " + std::to_string(r.search.nodes);
}

}  // namespace

TEST_CASE(gpu_abdada_one_worker_is_sequential) {
    // One GPU thread searches slowly: every position in the default
    // variant, three in the others.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const int depth = quick ? 3 : 4;
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Terms, EvalKind::Material}) {
            for (const FeatureKind f : {FeatureKind::None, FeatureKind::TTKillers, FeatureKind::TTKillersSee}) {
                for (const AttacksKind attacks : {AttacksKind::Hq, AttacksKind::Magic}) {
                    const GpuSearchOptions o = abdada_variant(board, eval, f, attacks);
                    if (!compiled(o)) continue;
                    // Quick: the default variant's neighbours only.
                    if (quick && (board != first_board() || (eval != EvalKind::Pesto && f != FeatureKind::None) ||
                                  (attacks != AttacksKind::Hq && f != FeatureKind::TTKillers)))
                        continue;
                    GpuAbdadaDevice device(gpu_test_env::context(), o, 12);
                    const bool table = uses_tt(f);
                    const bool main_variant = board == first_board() && eval == EvalKind::Pesto &&
                                              attacks == AttacksKind::Hq;
                    const size_t count = quick ? 2 : main_variant ? fens.size() : 3;
                    // Stopped workers leave claims behind: the next search
                    // must not see them.
                    dtt.clear();
                    device.set_tt(dtt.desc());
                    device.search(testutil::parse_or_die(fens[0]), GameHistory{}, 3, AbdadaDeviceOptions{256, 1});
                    for (size_t i = 0; i < count; ++i) {
                        const Position p = testutil::parse_or_die(fens[i]);
                        // The full window, then zero windows on both sides of
                        // the score (the root window reaches the workers).
                        dtt.clear();
                        htt.clear();
                        const CpuSearchOptions c = sequential(board, eval, f, table ? &htt : nullptr);
                        const FixedDepthResult full =
                            cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, c);
                        const int v = full.search.score;
                        for (const auto& [alpha, beta] : {std::pair{-kScoreInfinite, kScoreInfinite},
                                                          std::pair{v - 1, v}, std::pair{v, v + 1}}) {
                            dtt.clear();
                            htt.clear();
                            device.set_tt(dtt.desc());
                            const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, alpha, beta, c);
                            AbdadaDeviceStats st;
                            const FixedDepthResult r = device.search(p, GameHistory{}, depth,
                                                                     AbdadaDeviceOptions{1, 1}, {}, &st, alpha, beta);
                            CHECK_MSG(!r.search.aborted && r.search.score == seq.search.score &&
                                          r.search.best == seq.search.best && r.search.nodes == seq.search.nodes &&
                                          st.deferred == 0 && st.workers == 1 && st.started == 1 &&
                                          st.finished == 1 && st.id_sum == 1 && device.claims() == 0,
                                      cb::describe(o) + " " + fens[i] + " window " + std::to_string(alpha) + " " +
                                          std::to_string(beta) + ": " + result_text(r) + " vs " + result_text(seq) +
                                          " (" + describe(st) + ", " + std::to_string(device.claims()) + " claims)");
                            ++checked;
                        }
                        // A PV hint (a legal move that isn't the best, so it
                        // changes the order) reaches the workers.
                        MoveList legal;
                        generate_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks(), legal);
                        if (!main_variant || legal.size < 2) continue;
                        dtt.clear();
                        htt.clear();
                        device.set_tt(dtt.desc());
                        const Move other = legal.moves[legal.size - 1] != full.search.best ? legal.moves[legal.size - 1]
                                                                                           : legal.moves[0];
                        const std::vector<Move> hint = {other};
                        const FixedDepthResult seq =
                            cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, c, hint);
                        const FixedDepthResult r =
                            device.search(p, GameHistory{}, depth, AbdadaDeviceOptions{1, 1}, hint);
                        CHECK_MSG(r.search.score == seq.search.score && r.search.best == seq.search.best &&
                                      r.search.nodes == seq.search.nodes,
                                  cb::describe(o) + " " + fens[i] + " with a PV hint: " + result_text(r) + " vs " +
                                      result_text(seq));
                    }
                }
            }
        }
    }
    CHECK(checked > 0);
}

TEST_CASE(gpu_abdada_workers_find_the_sequential_score) {
    // Without a table every worker's search is exact, so whichever finishes
    // first has the sequential score, and its best move scores the same.
    const bool quick = gpu_test_env::quick();
    const GpuSearchOptions o = abdada_variant(first_board(), EvalKind::Pesto, FeatureKind::None, AttacksKind::Hq);
    if (!compiled(o)) return;
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : fens.size();
    struct Setting {
        unsigned workers;
        int lanes;
        int depth;  // two single threads are slow
        AbdadaParams params;
    };
    const std::vector<Setting> settings =
        quick ? std::vector<Setting>{{64, 1, 3, {}}, {256, 32, 3, {1, 1}}}
              : std::vector<Setting>{{2, 1, 4, {}},      {64, 1, 5, {}},      {2048, 1, 5, {}}, {2048, 32, 5, {}},
                                     {512, 1, 5, {1, 0}}, {512, 1, 5, {3, 0}}, {512, 1, 5, {2, 1}},
                                     {99, 1, 5, {}}};  // 99 warps: a last block only partly used
    GpuAbdadaDevice device(gpu_test_env::context(), o, 16);
    unsigned long long deferred = 0;
    unsigned long long big_workers = 0, big_finished = 0;  // searches of 32 polls or more per worker
    unsigned long long big_nodes = 0, big_deferred = 0;
    for (size_t i = 0; i < count; ++i) {
        const Position p = testutil::parse_or_die(fens[i]);
        for (const Setting& s : settings) {
            const int depth = s.depth;
            const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                          sequential(o.board, o.eval, o.features, nullptr));
            AbdadaDeviceStats st;
            const FixedDepthResult r =
                device.search(p, GameHistory{}, depth, AbdadaDeviceOptions{s.workers, s.lanes, s.params}, {}, &st);
            const std::string where = fens[i] + " workers " + std::to_string(s.workers) + " lanes " +
                                      std::to_string(s.lanes) + " depth " + std::to_string(depth) +
                                      describe(s.params) + " (" + describe(st) + ")";
            REQUIRE_MSG(!r.search.aborted, where);
            // Every worker runs once, with its own number, and at least one
            // finishes. The first to finish stops the others at their next
            // poll (every 1024 nodes): in trees big enough for many polls
            // most never finish (below, over all of them); in small ones
            // every worker finishes before it looks.
            CHECK_MSG(st.workers == s.workers && st.started == s.workers &&
                          st.id_sum == (unsigned long long)s.workers * (s.workers + 1) / 2 && st.finished >= 1,
                      where);
            if (s.workers >= 64 && seq.search.nodes >= 32 * kStopPollInterval) {
                big_workers += s.workers;
                big_finished += st.finished;
                big_nodes += st.nodes;
                big_deferred += st.deferred;
            }
            CHECK_MSG(r.search.score == seq.search.score,
                      where + ": " + std::to_string(r.search.score) + " vs " + std::to_string(seq.search.score));
            if (seq.search.best.is_none()) {
                CHECK_MSG(r.search.best.is_none(), where);
            } else {
                REQUIRE_MSG(is_legal_move(p, r.search.best), where + ": best " + move_to_uci(r.search.best));
                const int v = gpu_search_util::score_of_move(p, r.search.best, depth, o.eval);
                CHECK_MSG(v == seq.search.score,
                          where + ": best " + move_to_uci(r.search.best) + " scores " + std::to_string(v));
            }
            CHECK_MSG(r.pv.size() == (r.search.best.is_none() ? 0u : 1u) &&
                          (r.pv.empty() || r.pv.front() == r.search.best),
                      where + ": the PV is the best move");
            deferred += st.deferred;
        }
    }
    CHECK_MSG(deferred > 0, "no move was ever deferred");
    CHECK_MSG(quick || (big_workers >= 1000 && big_finished * 10 < big_workers * 9),
              std::to_string(big_finished) + " of " + std::to_string(big_workers) +
                  " workers finished in the big searches");
    // Workers sharing the busy table defer moves all the time (0.5% of the
    // nodes here, where every worker searches the whole tree without a
    // table; over 10% in small trees); a table that claims and checks
    // different slots defers next to none.
    CHECK_MSG(quick || big_deferred * 1000 >= big_nodes,
              std::to_string(big_deferred) + " moves deferred in " + std::to_string(big_nodes) + " nodes");
}

TEST_CASE(gpu_abdada_with_a_table) {
    // Iterative deepening on one shared table, as the engine searches: every
    // iteration ends with a legal move, and at the last depth most scores
    // equal the sequential search's (the table lets workers reuse each
    // other's deeper results, as any search with a table may).
    const bool quick = gpu_test_env::quick();
    const GpuSearchOptions o =
        abdada_variant(first_board(), EvalKind::Pesto, FeatureKind::TTKillers, AttacksKind::Hq);
    if (!compiled(o)) return;
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : fens.size();
    const int depth = quick ? 4 : 6;
    GpuAbdadaDevice device(gpu_test_env::context(), o, 18);
    DeviceTT dtt(64);
    HostTT htt(64);
    int same = 0, total = 0;
    unsigned long long deferred = 0;
    for (size_t i = 0; i < count; ++i) {
        const Position p = testutil::parse_or_die(fens[i]);
        const bool has_moves = count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks()) > 0;
        dtt.clear();
        device.set_tt(dtt.desc());
        std::vector<Move> pv;
        FixedDepthResult r;
        for (int d = 1; d <= depth; ++d) {
            dtt.new_search();
            AbdadaDeviceStats st;
            r = device.search(p, GameHistory{}, d, AbdadaDeviceOptions{2048, 1}, pv, &st);
            const std::string where = fens[i] + " depth " + std::to_string(d) + " (" + describe(st) + ")";
            REQUIRE_MSG(!r.search.aborted, where);
            CHECK_MSG(has_moves ? is_legal_move(p, r.search.best) : r.search.best.is_none(),
                      where + ": best " + move_to_uci(r.search.best));
            deferred += st.deferred;
            pv = r.pv;
        }
        htt.clear();
        const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                      sequential(o.board, o.eval, o.features, &htt));
        same += r.search.score == seq.search.score ? 1 : 0;
        ++total;
    }
    CHECK_MSG(same * 4 >= total * 3, std::to_string(same) + " of " + std::to_string(total) + " scores as sequential");
    CHECK_MSG(deferred > 0, "no move was ever deferred");
}

TEST_CASE(gpu_abdada_stops_and_caps) {
    const GpuSearchOptions o = abdada_variant(first_board(), EvalKind::Pesto, FeatureKind::None, AttacksKind::Hq);
    if (!compiled(o)) return;
    const Position p = testutil::parse_or_die(positions()[0]);
    GpuAbdadaDevice device(gpu_test_env::context(), o, 12);
    // A stop requested before the search, then a deadline already past: both
    // end it at once, reported as aborted.
    device.request_stop();
    CHECK(device.search(p, GameHistory{}, 12, AbdadaDeviceOptions{256, 1}).search.aborted);
    device.clear_stop();
    device.set_deadline(std::chrono::steady_clock::now());
    CHECK(device.search(p, GameHistory{}, 12, AbdadaDeviceOptions{256, 1}).search.aborted);
    device.set_deadline(std::chrono::steady_clock::time_point::max());
    // Too many workers: as many as fit, and the search still finishes.
    for (const int lanes : {1, 32}) {
        AbdadaDeviceStats st;
        const FixedDepthResult r = device.search(p, GameHistory{}, 3, AbdadaDeviceOptions{1u << 20, lanes}, {}, &st);
        CHECK_MSG(!r.search.aborted && st.workers == device.max_workers(lanes) && st.workers < (1u << 20),
                  describe(st));
    }
    CHECK(device.max_workers(32) == 32 * device.max_workers(1));
}

TEST_CASE(gpu_abdada_game_history) {
    // Repetitions through the game's positions: one worker exactly as the
    // CPU, several workers the same score (no table).
    const GpuSearchOptions o = abdada_variant(first_board(), EvalKind::Pesto, FeatureKind::None, AttacksKind::Hq);
    if (!compiled(o)) return;
    struct Game {
        const char* fen;
        const char* moves;
    };
    const Game games[] = {
        {"6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1"},
        {"7k/8/8/8/8/1q6/8/K2Q4 w - - 0 1", "d1h5 h8g7 h5g5 g7h7 g5h5 h7g7 h5g5 g7h7"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3 g8f6 f3g1 f6g8"},
    };
    GpuAbdadaDevice device(gpu_test_env::context(), o, 12);
    const int max_depth = gpu_test_env::quick() ? 3 : 5;
    for (const Game& g : games) {
        Position root;
        GameHistory history;
        std::string error;
        REQUIRE_MSG(build_game(testutil::parse_or_die(g.fen), g.moves, root, history, error), error);
        for (int depth = 1; depth <= max_depth; ++depth) {
            const FixedDepthResult seq = cpu_search_fixed(root, history, depth, -kScoreInfinite, kScoreInfinite,
                                                          sequential(o.board, o.eval, o.features, nullptr));
            const FixedDepthResult one = device.search(root, history, depth, AbdadaDeviceOptions{1, 1});
            const FixedDepthResult many = device.search(root, history, depth, AbdadaDeviceOptions{512, 1});
            const std::string where = std::string(g.moves) + " depth " + std::to_string(depth);
            CHECK_MSG(one.search.score == seq.search.score && one.search.best == seq.search.best &&
                          one.search.nodes == seq.search.nodes,
                      where + ": " + result_text(one) + " vs " + result_text(seq));
            CHECK_MSG(many.search.score == seq.search.score,
                      where + ": " + result_text(many) + " vs " + result_text(seq));
        }
    }
}
