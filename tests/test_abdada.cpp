// ABDADA (core/abdada.h; host/search_cpu.h cpu_search_abdada).
//
// What must hold:
// - a busy slot claims a position for one worker: other workers see it busy,
//   the claiming worker and other positions don't;
// - a move the list picker gets back comes after every other move, and is
//   recognized as deferred when it comes;
// - one worker searches exactly like the sequential search with list
//   ordering (score, best move, nodes, PV): nothing is ever busy for it;
// - several workers return the sequential score (without a table, where
//   every worker's search is exact) and a best move worth it, and do defer
//   moves, under every coordination setting (AbdadaParams); with a table, a
//   legal move and a score near the sequential one.

#include <memory>
#include <string>
#include <vector>

#include "core/abdada.h"
#include "core/search.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "recursive_search_ref.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;
using testref::attacks;

namespace {

#ifdef NDEBUG
constexpr int kDepth = 5;
#else
constexpr int kDepth = 3;
#endif

CpuSearchOptions list_options(FeatureKind f = FeatureKind::None) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::List};
    o.features = f;
    return o;
}

bool is_legal(const Position& p, Move m) {
    MoveList list;
    generate_legal(to_board<QuadBoard>(p), attacks(), list);
    for (int i = 0; i < list.size; ++i)
        if (list.moves[i] == m) return true;
    return false;
}

}  // namespace

TEST_CASE(abdada_busy_slots) {
    const u64 key = 0x123456789ABCDEF0ull;
    const u64 slot = abdada::slot_value(key, 3);
    CHECK(abdada::busy_for(slot, key, 4));   // another worker
    CHECK(!abdada::busy_for(slot, key, 3));  // its own claim
    CHECK(!abdada::busy_for(slot, key ^ (u64(1) << 40), 4));  // another position in the same slot
    CHECK(!abdada::busy_for(0, key, 4));     // an empty slot
    CHECK(abdada::busy_for(abdada::slot_value(key, 0), key, 1));  // worker 0's claims aren't empty
    CHECK((slot & abdada::kWorkerMask) == 4);
}

TEST_CASE(abdada_list_picker_defers) {
    const Position p = testutil::parse_or_die("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8");
    const QuadBoard b = to_board<QuadBoard>(p);
    const auto arena = std::make_unique<MoveArena<4096>>();
    ListPicker::State st;
    ListPicker::init(st, *arena, 0, b, attacks(), PickMode::All, Move());
    const int total = st.end - st.begin;
    REQUIRE(total > 10);
    const Move first = ListPicker::next(st, *arena);
    CHECK(!ListPicker::returned_deferred(st, *arena));
    ListPicker::defer(st, *arena);  // the first move goes back
    std::vector<Move> rest;
    for (Move m = ListPicker::next(st, *arena); !m.is_none(); m = ListPicker::next(st, *arena)) {
        rest.push_back(m);
        const bool deferred = ListPicker::returned_deferred(st, *arena);
        CHECK_MSG(deferred == (m == first), move_to_uci(m));
    }
    REQUIRE(int(rest.size()) == total);
    CHECK(rest.back() == first);  // after every other move
}

TEST_CASE(abdada_one_worker_is_sequential) {
    AbdadaTable table(12);
    HostTT tt_a(16), tt_b(16);
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        for (const FeatureKind f : {FeatureKind::None, FeatureKind::TTKillers}) {
            CpuSearchOptions o = list_options(f);
            tt_a.clear();
            tt_b.clear();
            o.tt = tt_a.desc();
            const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o);
            o.tt = tt_b.desc();
            AbdadaStats st;
            // Every move first claimed by this one worker: nothing to defer,
            // even with the first move deferrable.
            const FixedDepthResult r = cpu_search_abdada(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o, {},
                                                         1, table, nullptr, std::chrono::steady_clock::time_point::max(),
                                                         &st, AbdadaParams{1, 1});
            CHECK_MSG(r.search.score == seq.search.score && r.search.best == seq.search.best &&
                          r.search.nodes == seq.search.nodes && r.pv == seq.pv && st.deferred == 0 &&
                          table.claims() == 0,
                      fen + " " + name_of(f) + ": " + std::to_string(r.search.score) + "/" +
                          std::to_string(r.search.nodes) + " vs " + std::to_string(seq.search.score) + "/" +
                          std::to_string(seq.search.nodes));
        }
    }
}

TEST_CASE(abdada_workers_find_the_sequential_score) {
    // Without a table every worker's search is exact, so whichever finishes
    // first has the sequential score; its best move scores the same.
    AbdadaTable table(16);
    u64 deferred = 0;
    int searches = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const CpuSearchOptions o = list_options();
        const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o);
        struct Setting {
            int workers;
            AbdadaParams params;
        };
        for (const Setting& s : {Setting{2, {}}, Setting{4, {}}, Setting{8, {}}, Setting{4, {1, 0}}, Setting{4, {3, 0}},
                                 Setting{4, {2, 1}}, Setting{8, {1, 1}}}) {
            const int workers = s.workers;
            AbdadaStats st;
            const FixedDepthResult r = cpu_search_abdada(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o,
                                                         {}, workers, table, nullptr,
                                                         std::chrono::steady_clock::time_point::max(), &st, s.params);
            const std::string where = fen + " workers " + std::to_string(workers) + describe(s.params);
            REQUIRE_MSG(!r.search.aborted, where);
            CHECK_MSG(r.search.score == seq.search.score,
                      where + ": " + std::to_string(r.search.score) + " vs " + std::to_string(seq.search.score));
            if (seq.search.best.is_none()) {
                CHECK(r.search.best.is_none());
            } else {
                REQUIRE_MSG(is_legal(p, r.search.best), where + ": best " + move_to_uci(r.search.best));
                // The best move's own score is the root's.
                QuadBoard child = to_board<QuadBoard>(p);
                make_move(child, r.search.best);
                GameHistory h;
                h.entries.push_back(HistoryEntry{compute_key(to_board<QuadBoard>(p), attacks(), testref::tables().zobrist),
                                                 -1, 0, 0, 0});
                const int from_child =
                    -cpu_search_fixed(to_position(child), h, kDepth - 1, -kScoreInfinite, kScoreInfinite, o).search.score;
                // Mate scores count plies from the search's root: one more from the root's.
                const int v = is_mate_score(from_child) ? from_child - (from_child > 0 ? 1 : -1) : from_child;
                CHECK_MSG(v == seq.search.score, where + ": best " + move_to_uci(r.search.best) + " scores " +
                                                     std::to_string(v));
            }
            deferred += st.deferred;
            ++searches;
        }
    }
    CHECK_MSG(deferred > 0, "no move was ever deferred in " + std::to_string(searches) + " searches");
}

TEST_CASE(abdada_workers_with_a_table) {
    // Sharing a table, workers take each other's results, so the first to
    // finish may differ from the sequential search by the usual table effects
    // (deeper results reused): a legal best move, and the same score in most
    // positions.
    AbdadaTable table(16);
    HostTT tt(64);
    int same = 0, total = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        CpuSearchOptions o = list_options(FeatureKind::TTKillers);
        tt.clear();
        o.tt = tt.desc();
        const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o);
        tt.clear();
        AbdadaStats st;
        const FixedDepthResult r = cpu_search_abdada(p, GameHistory{}, kDepth, -kScoreInfinite, kScoreInfinite, o, {}, 8,
                                                     table, nullptr, std::chrono::steady_clock::time_point::max(), &st);
        REQUIRE_MSG(!r.search.aborted, fen);
        CHECK_MSG(seq.search.best.is_none() ? r.search.best.is_none() : is_legal(p, r.search.best),
                  fen + ": best " + move_to_uci(r.search.best));
        same += r.search.score == seq.search.score ? 1 : 0;
        ++total;
    }
    CHECK_MSG(same * 4 >= total * 3, std::to_string(same) + " of " + std::to_string(total) + " scores as sequential");
}

TEST_CASE(abdada_stops) {
    // A stop request ends every worker; the search reports itself aborted.
    AbdadaTable table(12);
    const Position p = testutil::parse_or_die(testutil::test_fens()[0]);
    std::atomic<bool> stop{true};
    const FixedDepthResult r = cpu_search_abdada(p, GameHistory{}, 9, -kScoreInfinite, kScoreInfinite, list_options(),
                                                 {}, 4, table, &stop);
    CHECK(r.search.aborted);
}

TEST_CASE(abdada_settings) {
    CHECK(describe(AbdadaParams{}).empty());
    CHECK_EQ(describe(AbdadaParams{3, 1}), std::string(" from depth 3 defer first"));
    for (const AbdadaParams bad : {AbdadaParams{0, 0}, AbdadaParams{21, 0}, AbdadaParams{2, 2}}) {
        bool threw = false;
        try {
            validate_abdada(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK_MSG(threw, describe(bad));
    }
}
