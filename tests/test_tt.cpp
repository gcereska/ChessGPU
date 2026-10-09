// The transposition table (core/tt.h): entry encoding, both layouts'
// probe, store and replacement rules, and the CPU search with a table.
//
// What the search must guarantee:
// - tt-order (the table only orders moves) never changes a score.
// - With cutoffs, scores equal the search without a table up to depth 3
//   plus quiescence: there no position recurs at another distance from the
//   root (that takes 4 plies) and none repeats, so every cutoff reuses a
//   result of exactly the depth needed.

#include <string>
#include <vector>

#include "core/tt.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/position.h"
#include "host/search_cpu.h"
#include "host/task_runner.h"
#include "host/tt_storage.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

#ifdef NDEBUG
constexpr int kOrderDepth = 4;
constexpr long long kTreeBudget = 6;
#else
constexpr int kOrderDepth = 3;
constexpr long long kTreeBudget = 1;
#endif

// The test positions plus positions from their trees.
std::vector<Position> tt_test_positions() {
    std::vector<Position> roots;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = kTreeBudget;
        auto visit = [&](const Position& p, const std::vector<Move>&) { roots.push_back(p); };
        testutil::walk_tree(testutil::parse_or_die(fen), 2, budget, visit);
    }
    return roots;
}

CpuSearchOptions with_tt(FeatureKind f, TTLayoutKind layout, const HostTT& table, EvalKind eval = EvalKind::Pesto) {
    CpuSearchOptions o{BoardKind::Quad, eval, OrderingKind::Staged};
    o.features = f;
    o.tt_layout = layout;
    o.tt = table.desc();
    return o;
}

FixedDepthResult search(const Position& p, int depth, const CpuSearchOptions& o) {
    return cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, o);
}

// The score `move` gets at `depth` from p, searched without a table (the
// child searched as a root of its own, at depth - 1: quiescence for depth 1).
int score_of_move(const Position& p, Move m, int depth, EvalKind eval) {
    QuadBoard b = to_board<QuadBoard>(p);
    make_move(b, m);
    const int child = cpu_search_fixed(to_position(b), GameHistory{}, depth - 1, -kScoreInfinite, kScoreInfinite,
                                       {BoardKind::Quad, eval, OrderingKind::Staged})
                          .search.score;
    int s = -child;
    // A mate seen from the child is one ply further from p.
    if (s > kScoreMateBound) s -= 1;
    if (s < -kScoreMateBound) s += 1;
    return s;
}

const TTLayoutKind kLayouts[] = {TTLayoutKind::Packed64, TTLayoutKind::Wide128};

std::string where(const Position& p, int depth, FeatureKind f, TTLayoutKind l) {
    return format_fen(p) + " depth " + std::to_string(depth) + " " + name_of(f) + " " + name_of(l);
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding and rules
// ---------------------------------------------------------------------------

TEST_CASE(tt_data_round_trip) {
    const Move moves[] = {Move(), Move(E2, E4, 0), Move(A7, A8, kPromotionFlagBit | 3), Move(H1, H8, kCaptureFlagBit)};
    const int scores[] = {0, 1, -1, 35, -1200, kScoreMate, -kScoreMate, kScoreMateBound + 1, 32767, -32768};
    for (const Move m : moves)
        for (const int s : scores)
            for (const int depth : {0, 1, 7, 63})
                for (const TTBound b : {TTBound::Upper, TTBound::Lower, TTBound::Exact})
                    for (const u32 g : {0u, 1u, 33u, 63u}) {
                        const TTData d = TTData::make(m, s, depth, b, g);
                        CHECK(d.hit());
                        CHECK(d.move() == m);
                        CHECK_EQ(d.score(), s);
                        CHECK_EQ(d.depth(), depth);
                        CHECK(d.bound() == b);
                        CHECK_EQ(d.generation(), g);
                        CHECK_EQ(d.raw & ~kTTDataMask, u64(0));
                        CHECK(d.with_move(Move(B1, C3, 0)).move() == Move(B1, C3, 0));
                        CHECK_EQ(d.with_move(m).raw, d.raw);
                    }
    CHECK(!TTData{}.hit());
    CHECK_EQ(TTData::make(Move(), 0, 0, TTBound::Upper, 64).generation(), 0u);  // generations wrap
}

TEST_CASE(tt_mate_scores_count_from_the_node) {
    // Mate 5 plies from the root, found at ply 2: 3 plies from that node.
    const int stored = tt_score_to(mate_in(5), 2);
    CHECK_EQ(stored, mate_in(3));
    // The same position met at ply 4: still mate 3 plies away, 7 from the root.
    CHECK_EQ(tt_score_from(stored, 4), mate_in(7));
    CHECK_EQ(tt_score_from(tt_score_to(mated_in(6), 3), 1), mated_in(4));
    for (const int s : {0, 150, -150, kScoreMateBound, -kScoreMateBound})
        for (const int ply : {0, 1, 30})
            CHECK_EQ(tt_score_to(s, ply), s);  // other scores are absolute
    for (int ply = 0; ply < kMaxPly; ++ply)
        for (int m = ply; m < kMaxPly; ++m) {
            CHECK_EQ(tt_score_from(tt_score_to(mate_in(m), ply), ply), mate_in(m));
            CHECK_EQ(tt_score_from(tt_score_to(mated_in(m), ply), ply), mated_in(m));
        }
}

TEST_CASE(tt_bounds_and_cutoffs) {
    CHECK(tt_bound_of(50, 10, 50) == TTBound::Lower);
    CHECK(tt_bound_of(10, 10, 50) == TTBound::Upper);
    CHECK(tt_bound_of(11, 10, 50) == TTBound::Exact);
    CHECK(tt_cuts(TTBound::Exact, 0, -100, 100));
    CHECK(tt_cuts(TTBound::Lower, 100, -100, 100));
    CHECK(!tt_cuts(TTBound::Lower, 99, -100, 100));
    CHECK(tt_cuts(TTBound::Upper, -100, -100, 100));
    CHECK(!tt_cuts(TTBound::Upper, -99, -100, 100));
    CHECK(!tt_cuts(TTBound::None, 0, -100, 100));
}

TEST_CASE(tt_table_sizes_and_generations) {
    CHECK_EQ(tt_bucket_count(1), u64(1) << 15);   // 1 MiB of 32-byte buckets
    CHECK_EQ(tt_bucket_count(3), u64(1) << 16);   // rounded down to a power of two
    CHECK_EQ(tt_bucket_count(0), u64(1) << 15);   // at least 1 MiB
    CHECK_EQ(tt_bucket_count(2048), u64(1) << 26);
    HostTT t(2);
    CHECK_EQ(t.bucket_count(), u64(1) << 16);
    CHECK_EQ(t.desc().bucket_mask, (u64(1) << 16) - 1);
    CHECK_EQ(t.size_mb(), size_t(2));
    for (u32 i = 1; i <= kTTGenerations; ++i) {
        t.new_search();
        CHECK_EQ(t.generation(), i % kTTGenerations);
    }
    TTView<TTPacked64> v(t.desc());
    v.store(12345, TTData::make(Move(E2, E4, 0), 10, 3, TTBound::Exact, t.generation()));
    CHECK(v.probe(12345).hit());
    t.clear();
    CHECK(!v.probe(12345).hit());
    t.resize(1);
    CHECK_EQ(t.bucket_count(), u64(1) << 15);
}

namespace {

// Keys that land in bucket 0 of a table of `buckets` buckets, distinct top bits.
u64 key_in_bucket0(int i) { return (u64(i + 1) << 48) | (u64(i + 1) << 20); }

template <class Layout>
void check_layout() {
    HostTT table(1);
    const u32 gen = 5;
    TTDesc desc = table.desc();
    desc.generation = gen;
    const TTView<Layout> tt(desc);
    const u64 k = key_in_bucket0(0);
    const Move m1(E2, E4, 0), m2(D2, D4, 0);

    // Store and probe; another key in the same bucket misses.
    tt.store(k, TTData::make(m1, 42, 5, TTBound::Exact, gen));
    TTData e = tt.probe(k);
    CHECK(e.hit() && e.move() == m1 && e.score() == 42 && e.depth() == 5 && e.bound() == TTBound::Exact);
    CHECK_EQ(e.generation(), gen);
    CHECK(!tt.probe(key_in_bucket0(1)).hit());

    // Same position: a much shallower bound from this search doesn't replace...
    tt.store(k, TTData::make(m2, 7, 1, TTBound::Lower, gen));
    CHECK_EQ(tt.probe(k).depth(), 5);
    // ...a slightly shallower one does, and keeps the old move if it has none.
    tt.store(k, TTData::make(Move(), 9, 2, TTBound::Upper, gen));
    e = tt.probe(k);
    CHECK(e.depth() == 2 && e.score() == 9 && e.move() == m1);
    // An exact result always replaces; so does any result over an old entry.
    tt.store(k, TTData::make(m2, 11, 9, TTBound::Lower, gen));
    tt.store(k, TTData::make(m1, 12, 1, TTBound::Exact, gen));
    CHECK_EQ(tt.probe(k).depth(), 1);
    {
        TTDesc later = desc;
        later.generation = gen + 1;
        const TTView<Layout> next(later);
        tt.store(k, TTData::make(m1, 13, 30, TTBound::Lower, gen));
        next.store(k, TTData::make(m2, 14, 1, TTBound::Upper, gen + 1));
        CHECK_EQ(next.probe(k).score(), 14);
    }
    table.clear();

    // A full bucket: the new entry takes the slot of the least worth (the
    // shallowest of this search; entries of older searches before deeper ones).
    for (int i = 0; i < Layout::kEntries; ++i)
        tt.store(key_in_bucket0(i), TTData::make(m1, i, 10 + i, TTBound::Exact, gen));
    for (int i = 0; i < Layout::kEntries; ++i) CHECK(tt.probe(key_in_bucket0(i)).hit());
    tt.store(key_in_bucket0(10), TTData::make(m2, 99, 3, TTBound::Lower, gen));
    CHECK(tt.probe(key_in_bucket0(10)).hit());
    CHECK(!tt.probe(key_in_bucket0(0)).hit());  // depth 10, the shallowest
    for (int i = 1; i < Layout::kEntries; ++i) CHECK(tt.probe(key_in_bucket0(i)).hit());
    table.clear();
    {
        TTDesc old = desc;
        old.generation = gen - 2;
        const TTView<Layout> before(old);
        before.store(key_in_bucket0(0), TTData::make(m1, 1, 30, TTBound::Exact, gen - 2));  // deep but 2 searches old
        for (int i = 1; i < Layout::kEntries; ++i)
            tt.store(key_in_bucket0(i), TTData::make(m1, 1, 20, TTBound::Exact, gen));
        tt.store(key_in_bucket0(20), TTData::make(m2, 2, 1, TTBound::Upper, gen));
        CHECK(!tt.probe(key_in_bucket0(0)).hit());  // worth 30 - 2 * 8 = 14, below 20
        CHECK(tt.probe(key_in_bucket0(20)).hit());
    }
}

}  // namespace

TEST_CASE(tt_packed64_rules) { check_layout<TTPacked64>(); }
TEST_CASE(tt_wide128_rules) { check_layout<TTWide128>(); }

TEST_CASE(tt_wide128_rejects_torn_entries) {
    HostTT table(1);
    const TTView<TTWide128> tt(table.desc());
    const u64 ka = key_in_bucket0(0), kb = key_in_bucket0(1);
    const TTData da = TTData::make(Move(E2, E4, 0), 1, 4, TTBound::Exact, 0);
    const TTData db = TTData::make(Move(D2, D4, 0), 2, 5, TTBound::Lower, 0);
    tt.store(ka, da);
    // Another writer stored b's key word into the slot, then was cut short.
    table.desc().buckets[0].w[0] = kb ^ db.raw;
    CHECK(!tt.probe(ka).hit());
    CHECK(!tt.probe(kb).hit());
    table.desc().buckets[0].w[1] = db.raw;  // the second word lands: b is complete
    CHECK(tt.probe(kb).hit());
}

TEST_CASE(tt_packed64_false_hit_is_possible) {
    // Two positions with the same bucket and the same top 16 bits look alike
    // to packed64 (about 1 in 16k probes of absent positions); wide128 tells
    // them apart. The search checks a stored move's legality, so a false hit
    // can only mis-score.
    HostTT table(1);
    const u64 a = (u64(7) << 48) | (u64(1) << 30), b = (u64(7) << 48) | (u64(2) << 30);
    const TTData d = TTData::make(Move(E2, E4, 0), 1, 4, TTBound::Exact, 0);
    TTView<TTPacked64>(table.desc()).store(a, d);
    CHECK(TTView<TTPacked64>(table.desc()).probe(b).hit());
    table.clear();
    TTView<TTWide128>(table.desc()).store(a, d);
    CHECK(!TTView<TTWide128>(table.desc()).probe(b).hit());
}

// ---------------------------------------------------------------------------
// Search with a table
// ---------------------------------------------------------------------------

TEST_CASE(tt_needs_a_table) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TT;
    bool threw = false;
    try {
        search(Position{testutil::parse_or_die(kStartFen)}, 2, o);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE(tt_order_only_keeps_every_score) {
    // Fresh tables and an iterative-deepening run sharing one table: the
    // table only reorders moves, so scores stay exact and the best move
    // scores the same.
    const std::vector<Position> roots = tt_test_positions();
    HostTT table(16);
    int moves_differ = 0;
    for (const Position& p : roots) {
        for (const TTLayoutKind layout : kLayouts) {
            table.clear();
            table.new_search();
            for (int depth = 1; depth <= kOrderDepth; ++depth) {
                const FixedDepthResult base = search(p, depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged});
                const FixedDepthResult got = search(p, depth, with_tt(FeatureKind::TTOrder, layout, table));
                CHECK_MSG(got.search.score == base.search.score,
                          where(p, depth, FeatureKind::TTOrder, layout) + ": " + std::to_string(got.search.score) +
                              " vs " + std::to_string(base.search.score));
                if (got.search.best != base.search.best && !got.search.best.is_none()) {
                    ++moves_differ;
                    CHECK_EQ(score_of_move(p, got.search.best, depth, EvalKind::Pesto), base.search.score);
                }
            }
        }
    }
    CHECK(roots.size() > testutil::test_fens().size());
    (void)moves_differ;
}

TEST_CASE(tt_move_ordering_saves_nodes) {
    // Iterative deepening with tt-order: the stored best moves are tried
    // first at every node (not only along the PV), so the trees shrink.
    HostTT table(16);
    u64 with = 0, without = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        table.clear();
        table.new_search();
        std::vector<Move> pv;
        std::vector<Move> pv_base;
        for (int depth = 1; depth <= kOrderDepth; ++depth) {
            const FixedDepthResult r = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                        with_tt(FeatureKind::TTOrder, TTLayoutKind::Packed64, table),
                                                        pv);
            const FixedDepthResult b = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                        {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged},
                                                        pv_base);
            with += r.search.nodes;
            without += b.search.nodes;
            pv = r.pv;
            pv_base = b.pv;
        }
    }
    CHECK_MSG(with < without, std::to_string(with) + " nodes ordered by the table vs " + std::to_string(without));
}

TEST_CASE(tt_cutoffs_exact_to_depth_3) {
    const std::vector<Position> roots = tt_test_positions();
    HostTT table(16);
    for (const Position& p : roots) {
        for (const FeatureKind f : {FeatureKind::TT, FeatureKind::TTNoQS}) {
            for (const TTLayoutKind layout : kLayouts) {
                for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
                    table.clear();  // one table through the iterations, like the engine
                    table.new_search();
                    FixedDepthResult base;
                    for (int depth = 1; depth <= 3; ++depth) {
                        base = search(p, depth, {BoardKind::Quad, eval, OrderingKind::Staged});
                        const FixedDepthResult got = search(p, depth, with_tt(f, layout, table, eval));
                        CHECK_MSG(got.search.score == base.search.score,
                                  where(p, depth, f, layout) + " " + name_of(eval) + ": " +
                                      std::to_string(got.search.score) + " vs " + std::to_string(base.search.score));
                    }
                    // The same search again: the root's own entry is stored now,
                    // but the root never cuts off (it must return a move).
                    const FixedDepthResult again = search(p, 3, with_tt(f, layout, table, eval));
                    CHECK_EQ(again.search.score, base.search.score);
                    CHECK_MSG(again.search.best.is_none() == base.search.best.is_none(),
                              where(p, 3, f, layout) + ": the repeated search lost its move");
                    // A fresh table for the deepest search too.
                    table.clear();
                    const FixedDepthResult fresh = search(p, 3, with_tt(f, layout, table, eval));
                    CHECK_EQ(fresh.search.score,
                             search(p, 3, {BoardKind::Quad, eval, OrderingKind::Staged}).search.score);
                }
            }
        }
    }
}

TEST_CASE(tt_variants_agree_and_prune) {
    // Every ordering and board give the same results with a table as with
    // each other (same tree, same table operations), and the table prunes.
    HostTT table(16);
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        for (const TTLayoutKind layout : kLayouts) {
            FixedDepthResult first;
            bool have = false;
            for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    table.clear();
                    CpuSearchOptions o = with_tt(FeatureKind::TT, layout, table);
                    o.board = board;
                    o.ordering = ordering;
                    const FixedDepthResult r = search(p, kOrderDepth, o);
                    if (!have) {
                        first = r;
                        have = true;
                        continue;
                    }
                    CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                                  r.search.best == first.search.best,
                              fen + " " + name_of(board) + "/" + name_of(ordering) + " " + name_of(layout));
                }
            }
        }
    }
    const Position start = testutil::parse_or_die(kStartFen);
    table.clear();
    const u64 with = search(start, 5, with_tt(FeatureKind::TT, TTLayoutKind::Packed64, table)).search.nodes;
    const u64 without = search(start, 5, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.nodes;
    CHECK_MSG(with < without, std::to_string(with) + " nodes with a table vs " + std::to_string(without));
}

TEST_CASE(tt_mate_distances_survive) {
    // Mates found through table entries stored at other plies keep their
    // length (scores are stored relative to the node).
    struct Case {
        const char* fen;
        int depth;
    };
    const Case cases[] = {
        {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3},              // mate in 1
        {"k7/8/1K6/8/8/8/8/7R b - - 0 1", 4},                  // mated in 1
        {"7k/8/6K1/8/8/8/8/6Q1 w - - 0 1", 5},                 // queen and king mate
        {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3},  // Qxf7#
        {"8/8/8/8/8/5k2/8/5K1R w - - 0 1", 5},                 // rook and king: no mate in reach
    };
    HostTT table(16);
    for (const Case& c : cases) {
        const Position p = testutil::parse_or_die(c.fen);
        for (const TTLayoutKind layout : kLayouts) {
            table.clear();
            table.new_search();
            for (int depth = 1; depth <= c.depth; ++depth) {
                const int base = search(p, depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.score;
                const int got = search(p, depth, with_tt(FeatureKind::TT, layout, table)).search.score;
                if (is_mate_score(base) || is_mate_score(got))
                    CHECK_MSG(got == base, std::string(c.fen) + " depth " + std::to_string(depth) + " " +
                                               name_of(layout) + ": " + std::to_string(got) + " vs " +
                                               std::to_string(base));
            }
        }
    }
}

TEST_CASE(tt_task_runner_shares_the_table) {
    // The CPU runner (the coordinators' test double) uses the table it is
    // given; a second run of the same tasks finds their results stored.
    HostTT table(16);
    CpuTaskRunner runner(with_tt(FeatureKind::TT, TTLayoutKind::Wide128, table));
    std::vector<TaskSpec> tasks;
    for (const std::string& fen : testutil::test_fens()) {
        TaskSpec t;
        t.board = to_board<QuadBoard>(testutil::parse_or_die(fen));
        t.depth = 3;
        tasks.push_back(t);
    }
    const std::vector<SearchResult> first = runner.run(tasks, {}, 0, {});
    const std::vector<SearchResult> second = runner.run(tasks, {}, 0, {});
    REQUIRE(first.size() == tasks.size() && second.size() == tasks.size());
    u64 nodes_first = 0, nodes_second = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        CHECK_EQ(second[i].score, first[i].score);
        // A task at ply 0 is the search root: it never cuts off, even though
        // its own entry is stored now (and these tasks aren't on a PV).
        CHECK_MSG(second[i].best.is_none() == first[i].best.is_none(), testutil::test_fens()[i]);
        nodes_first += first[i].nodes;
        nodes_second += second[i].nodes;
    }
    CHECK_MSG(nodes_second < nodes_first, std::to_string(nodes_second) + " vs " + std::to_string(nodes_first));
}
