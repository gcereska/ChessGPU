// Killer moves (core/move_order.h, core/search.h; feature sets tt-killers
// and tt-ext-killers).
//
// What must hold:
// - both pickers produce the same order with killers, each legal move once,
//   whatever the killers are: legal quiet moves, moves from other positions
//   (illegal here), captures, castling, duplicates, the PV move;
// - legal quiet killers come right after the captures and queen promotions,
//   the newer first;
// - the search with killers matches a recursive implementation exactly
//   (score and node count; without a table, through test-only feature sets),
//   alone and with check extensions;
// - killers save nodes; mates keep their distance; every board and ordering
//   visits the same tree.
//
// Of 15 bugs planted on the CPU side (the staged picker's killer stage and
// quiet stage, the order keys, the list picker, add_killer, and where the
// search reads and stores killers), all are caught; one (killers not checked
// for legality) by hanging, as illegal moves corrupt the board.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

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
constexpr int kRefDepth = 5;
constexpr int kNodesDepth = 6;
#else
constexpr int kRefDepth = 3;
constexpr int kNodesDepth = 5;
#endif

struct FeaturesKillersOnly {
    static constexpr const char* kName = "killers-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = true;
    static constexpr bool kQPrune = false;
};

struct FeaturesExtKillers {
    static constexpr const char* kName = "ext-killers";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = true;
    static constexpr bool kKillers = true;
    static constexpr bool kQPrune = false;
};

// Every technique at once, no table.
struct FeaturesEverything {
    static constexpr const char* kName = "everything";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = true;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = true;
    static constexpr bool kCheckExt = true;
    static constexpr bool kKillers = true;
    static constexpr bool kQPrune = false;
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

template <class Board>
std::vector<Move> staged_sequence(const Board& b, PickMode mode, Move pv, Killers k) {
    const CheckInfo ci = make_check_info(b, attacks());
    StagedPicker::State st;
    StagedPicker::init(st, b, attacks(), ci, mode, k);
    std::vector<Move> seq;
    for (Move m = StagedPicker::next(st, b, attacks(), ci, pv); !m.is_none(); m = StagedPicker::next(st, b, attacks(), ci, pv)) {
        seq.push_back(m);
        if (seq.size() > 300) break;  // runaway guard
    }
    return seq;
}

template <class Board>
std::vector<Move> list_sequence(const Board& b, PickMode mode, Move pv, Killers k) {
    const auto arena = std::make_unique<MoveArena<4096>>();
    ListPicker::State st;
    ListPicker::init(st, *arena, 100, b, attacks(), mode, pv, k);
    std::vector<Move> seq;
    for (Move m = ListPicker::next(st, *arena); !m.is_none(); m = ListPicker::next(st, *arena)) seq.push_back(m);
    return seq;
}

// Killer candidates for a position: its own quiet moves, moves of the other
// positions (mostly illegal here), captures, castling and the none move.
std::vector<Move> candidates(const Position& p, const std::vector<Move>& foreign) {
    const QuadBoard b = to_board<QuadBoard>(p);
    MoveList list;
    generate_legal(b, attacks(), list);
    std::vector<Move> own;
    for (int i = 0; i < list.size; ++i) own.push_back(list.moves[i]);
    std::vector<Move> c;
    for (size_t i = 0; i < own.size(); i += 3) c.push_back(own[i]);  // quiet and not: every third move
    for (size_t i = 0; i < foreign.size(); i += 7) c.push_back(foreign[i]);
    c.push_back(Move());
    return c;
}

template <class Board, class Features, bool kExt, bool kNull, bool kLmr, bool kFut>
void check_against_recursive() {
    const PruneParams prune;
    u64 updates = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kRefDepth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, PestoEval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + Features::kName;
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            updates += rec.killer_updates;
            for (const int alpha : {score - 1, score}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome zero = testref::flat_search<Board, PestoEval, Features>(
                    p, depth, alpha, alpha + 1, 0, 0, prune, pv);
                auto zrec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(zero.score == zscore && zero.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(zero.score) + "/" +
                              std::to_string(zero.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    CHECK_MSG(updates > 0, "no killer was ever stored");
}

CpuSearchOptions with_table(FeatureKind f, const HostTT& table, BoardKind board = BoardKind::Quad,
                            OrderingKind ordering = OrderingKind::Staged) {
    CpuSearchOptions o{board, EvalKind::Pesto, ordering};
    o.features = f;
    o.tt = table.desc();
    return o;
}

FixedDepthResult search(const Position& p, int depth, const CpuSearchOptions& o, const std::vector<Move>& pv = {}) {
    return cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, o, pv);
}

struct KillerSet {
    FeatureKind with;
    FeatureKind without;
};
const KillerSet kSets[] = {{FeatureKind::TTKillers, FeatureKind::TT}, {FeatureKind::TTExtKillers, FeatureKind::TTExt}};

}  // namespace

TEST_CASE(killers_order_key) {
    // A quiet killer moves to the killer group (the newer first); a killer
    // that isn't quiet keeps its own group; the PV move stays first.
    const Position p = testutil::parse_or_die("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8");
    const QuadBoard b = to_board<QuadBoard>(p);
    const Move quiet1 = parse_uci_move(b, attacks(), "a2a3");
    const Move quiet2 = parse_uci_move(b, attacks(), "f3e5");
    const Move capture = parse_uci_move(b, attacks(), "c4d5");
    const Killers k{quiet1, quiet2};
    CHECK_EQ(order_key(b, quiet1, Move(), k) >> 29, order::kKiller);
    CHECK_EQ(order_key(b, quiet2, Move(), k) >> 29, order::kKiller);
    CHECK(order_key(b, quiet1, Move(), k) > order_key(b, quiet2, Move(), k));
    CHECK_EQ(order_key(b, quiet1, Move(), {}) >> 29, order::kQuiet);
    const Killers kc{capture, quiet2};
    CHECK_EQ(order_key(b, capture, Move(), kc) >> 29, order::kCapture);
    CHECK_EQ(order_key(b, quiet1, quiet1, k), 0xFFFFFFFFu);
    // Killers rank above castling and quiet moves, below queen promotions.
    CHECK(order::kQueenPromotion > order::kKiller && order::kKiller > order::kCastle && order::kCastle > order::kQuiet);
}

TEST_CASE(killers_pickers_agree) {
    // Both pickers, every killer pair from the candidates, both modes (the
    // search uses PickMode::Quiescence only out of check and without a PV
    // move, as tests/test_search.cpp checks it).
    std::vector<Move> foreign;
    for (const std::string& fen : testutil::test_fens()) {
        const QuadBoard b = to_board<QuadBoard>(testutil::parse_or_die(fen));
        MoveList list;
        generate_legal(b, attacks(), list);
        for (int i = 0; i < list.size; ++i) foreign.push_back(list.moves[i]);
    }
    int pairs = 0, legal_killers = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        const QuadBoard qb = to_board<QuadBoard>(p);
        const StdBoard sb = to_board<StdBoard>(p);
        MoveList legal;
        generate_legal(qb, attacks(), legal);
        const bool check = make_check_info(qb, attacks()).in_check();
        const std::vector<Move> c = candidates(p, foreign);
        const Move pv = legal.size > 0 ? legal.moves[legal.size / 2] : Move();
        for (size_t i = 0; i < c.size(); ++i) {
            for (size_t j = 0; j < c.size(); j += 2) {
                const Killers k{c[i], c[j]};
                for (const PickMode mode : {PickMode::All, PickMode::Quiescence}) {
                    for (const Move hint : {Move(), pv}) {
                        if (mode == PickMode::Quiescence && (check || !hint.is_none())) continue;
                        const std::vector<Move> staged = staged_sequence(qb, mode, hint, k);
                        const std::string where = fen + " killers " + move_to_uci(k.first) + " " +
                                                  move_to_uci(k.second) + " pv " + move_to_uci(hint) +
                                                  (mode == PickMode::All ? " all" : " quiescence");
                        CHECK_MSG(staged == list_sequence(qb, mode, hint, k), where + ": list order differs");
                        CHECK_MSG(staged == staged_sequence(sb, mode, hint, k), where + ": std board differs");
                        // Each move once; in PickMode::All every legal move.
                        std::vector<Move> sorted = staged;
                        std::sort(sorted.begin(), sorted.end(), [](Move x, Move y) { return x.raw() < y.raw(); });
                        CHECK_MSG(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), where + ": twice");
                        if (mode == PickMode::All) CHECK_MSG(int(staged.size()) == legal.size, where + ": count");
                        // Order keys never increase along the sequence.
                        for (size_t n = 1; n < staged.size(); ++n)
                            CHECK_MSG(order_key(qb, staged[n - 1], hint, k) > order_key(qb, staged[n], hint, k),
                                      where + ": out of order at " + std::to_string(n));
                        if (mode == PickMode::All)
                            for (const Move m : staged)
                                if (order_key(qb, m, hint, k) >> 29 == order::kKiller) ++legal_killers;
                        ++pairs;
                    }
                }
            }
        }
    }
    CHECK(pairs > 1000 && legal_killers > 100);
}

TEST_CASE(killers_search_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesKillersOnly, false, false, false, false>();
    check_against_recursive<StdBoard, FeaturesKillersOnly, false, false, false, false>();
}

TEST_CASE(killers_with_extensions_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesExtKillers, true, false, false, false>();
}

TEST_CASE(killers_with_everything_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesEverything, true, true, true, true>();
}

TEST_CASE(killers_save_nodes) {
    // Iterative deepening on one table per position, as the engine searches.
    HostTT table(64);
    for (const KillerSet& set : kSets) {
        u64 with = 0, without = 0;
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            for (const FeatureKind f : {set.without, set.with}) {
                table.clear();
                std::vector<Move> pv;
                u64 total = 0;
                for (int depth = 1; depth <= kNodesDepth; ++depth) {
                    table.new_search();
                    const FixedDepthResult r = search(p, depth, with_table(f, table), pv);
                    total += r.search.nodes;
                    pv = r.pv;
                }
                (f == set.without ? without : with) += total;
            }
        }
        CHECK_MSG(with < without, std::string(name_of(set.with)) + ": " + std::to_string(with) +
                                      " nodes with killers vs " + std::to_string(without));
    }
}

TEST_CASE(killers_mates_keep_their_distance) {
    struct Case {
        const char* fen;
        int depth;
    };
    const Case cases[] = {
        {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3},
        {"k7/8/1K6/8/8/8/8/7R b - - 0 1", 4},
        {"7k/8/6K1/8/8/8/8/6Q1 w - - 0 1", 5},
        {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3},
    };
    HostTT table(16);
    for (const Case& c : cases) {
        const Position p = testutil::parse_or_die(c.fen);
        const int mate = search(p, c.depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.score;
        REQUIRE(is_mate_score(mate));
        for (const KillerSet& set : kSets) {
            table.clear();
            std::vector<Move> pv;
            int score = 0;
            for (int depth = 1; depth <= c.depth + 2; ++depth) {
                table.new_search();
                const FixedDepthResult r = search(p, depth, with_table(set.with, table), pv);
                score = r.search.score;
                pv = r.pv;
            }
            // Extensions may find a shorter mate, never a longer one.
            CHECK_MSG(is_mate_score(score) && score >= mate,
                      std::string(c.fen) + " " + name_of(set.with) + ": " + std::to_string(score) + " vs " +
                          std::to_string(mate));
        }
    }
}

TEST_CASE(killers_variants_agree) {
    HostTT table(16);
    for (const KillerSet& set : kSets) {
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            table.clear();
            const FixedDepthResult first = search(p, kRefDepth, with_table(set.with, table));
            for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    table.clear();
                    const FixedDepthResult r = search(p, kRefDepth, with_table(set.with, table, board, ordering));
                    CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                                  r.search.best == first.search.best && r.pv == first.pv,
                              fen + " " + name_of(set.with) + " " + name_of(board) + "/" + name_of(ordering));
                }
            }
        }
    }
}

TEST_CASE(killers_settings) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTKillers;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-killers wide128"));
    CHECK(uses_killers(FeatureKind::TTExtKillers) && uses_check_extensions(FeatureKind::TTExtKillers));
    CHECK(!uses_killers(FeatureKind::TTExt));
}

TEST_CASE(killers_update_and_packing) {
    // add_killer keeps the two latest distinct quiet cutoff moves, the newer
    // first; packing (the GPU tree's shared table) loses nothing.
    const Position p = testutil::parse_or_die("r3k2r/pPpp1ppp/8/3Pp3/8/8/PPP2PPP/R3K2R w KQkq e6 0 1");
    const QuadBoard b = to_board<QuadBoard>(p);
    const Move a = parse_uci_move(b, attacks(), "a2a3");
    const Move d = parse_uci_move(b, attacks(), "a2a4");
    const Move c = parse_uci_move(b, attacks(), "e1d1");
    REQUIRE(!a.is_none() && !d.is_none() && !c.is_none());
    Killers k{};
    k = add_killer(k, a);
    CHECK(k.first == a && k.second.is_none());
    k = add_killer(k, d);
    CHECK(k.first == d && k.second == a);
    k = add_killer(k, d);
    CHECK(k.first == d && k.second == a);
    k = add_killer(k, a);
    CHECK(k.first == a && k.second == d);
    k = add_killer(k, c);
    CHECK(k.first == c && k.second == a);
    const Killers u = unpack_killers(pack_killers(k));
    CHECK(u.first == k.first && u.second == k.second);
    CHECK_EQ(pack_killers(Killers{}), 0u);
    CHECK(unpack_killers(0).first.is_none() && unpack_killers(0).second.is_none());
    // Only quiet moves that aren't castling can be killers.
    CHECK(can_be_killer(a) && can_be_killer(d) && can_be_killer(c));
    for (const char* text : {"e1g1", "e1c1", "d5e6", "b7a8q", "b7b8n", "b7a8r"}) {
        const Move m = parse_uci_move(b, attacks(), text);
        REQUIRE_MSG(!m.is_none(), text);
        CHECK_MSG(!can_be_killer(m), text);
    }
}

TEST_CASE(killers_with_null_moves_and_reductions) {
    // The feature sets that try null moves and reductions again on top of
    // killers (their rules are checked exactly against the reference with
    // every technique at once, above): settings, mates, and the same tree on
    // every board and ordering.
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTKillersNmp;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-killers-nmp wide128 null 3+d/6 from 2"));
    o.features = FeatureKind::TTExtKillersLmr;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-ext-killers-lmr wide128 lmr 75/225 from 3 after 3"));
    CHECK(uses_null_move(FeatureKind::TTExtKillersNmp) && uses_check_extensions(FeatureKind::TTExtKillersNmp) &&
          uses_killers(FeatureKind::TTExtKillersNmp) && !uses_lmr(FeatureKind::TTExtKillersNmp));
    CHECK(uses_lmr(FeatureKind::TTKillersLmr) && !uses_null_move(FeatureKind::TTKillersLmr) &&
          !uses_check_extensions(FeatureKind::TTKillersLmr));
    HostTT table(16);
    const FeatureKind sets[] = {FeatureKind::TTKillersNmp, FeatureKind::TTKillersLmr, FeatureKind::TTExtKillersNmp,
                                FeatureKind::TTExtKillersLmr};
    for (const FeatureKind f : sets) {
        for (const char* fen : {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "7k/8/6K1/8/8/8/8/6Q1 w - - 0 1"}) {
            const Position p = testutil::parse_or_die(fen);
            table.clear();
            std::vector<Move> pv;
            int score = 0;
            for (int depth = 1; depth <= 7; ++depth) {
                table.new_search();
                const FixedDepthResult r = search(p, depth, with_table(f, table), pv);
                score = r.search.score;
                pv = r.pv;
            }
            CHECK_MSG(is_mate_score(score) && score > 0, std::string(fen) + " " + name_of(f) + ": " +
                                                             std::to_string(score));
        }
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            table.clear();
            const FixedDepthResult first = search(p, kRefDepth, with_table(f, table));
            for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    table.clear();
                    const FixedDepthResult r = search(p, kRefDepth, with_table(f, table, board, ordering));
                    CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                                  r.pv == first.pv,
                              fen + " " + name_of(f) + " " + name_of(board) + "/" + name_of(ordering));
                }
            }
        }
    }
}
