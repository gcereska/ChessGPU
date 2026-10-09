// NNUE (core/nnue.h, host/nnue_train.h):
// - input indices are distinct, in range, and mirror between the two sides;
// - with a random network: the incremental accumulator equals the one built
//   from scratch after every move and unmake restores it; a position and its
//   color-flipped twin score exactly alike; the integer evaluation stays
//   within rounding of the float network;
// - the trainer's gradient equals finite differences of the loss, for
//   inputs seen from either side, the biases and the output layer;
// - training on positions labeled by their material balance lowers the loss
//   and ranks material like the labels;
// - weights files round-trip, and wrong ones are refused; the C++ header the
//   trainer writes holds the same numbers.

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "core/nnue.h"
#include "core/search_board.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/nnue_train.h"
#include "host/position_moves.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

// A random float network (fixed seed) with a non-trivial output layer.
NnueFloat make_random_float_net(int hidden) {
    NnueFloat n = NnueFloat::random(7, hidden);
    std::mt19937_64 rng(11);
    std::normal_distribution<float> w2(0.0f, 0.4f), b1(0.3f, 0.2f);
    for (float& w : n.w2) w = w2(rng);
    for (float& b : n.b1) b = b1(rng);
    n.b2 = 0.05f;
    return n;
}

const NnueFloat& random_float_net() {
    static const NnueFloat net = make_random_float_net(kNnueHidden);
    return net;
}

// Search tables whose NNUE weights are random_float_net() quantized.
const SearchTables& random_tables() {
    static const std::unique_ptr<SearchTables> t = [] {
        auto p = std::make_unique<SearchTables>();
        init_search_tables(*p);
        p->eval.nnue = quantize_nnue(random_float_net());
        return p;
    }();
    return *t;
}

int nnue_eval(const Position& p, const EvalTables& t) {
    const QuadBoard b = to_board<QuadBoard>(p);
    return NnueEval::evaluate(compute_accumulator<NnueEval>(b, t), state::side(p.state));
}

std::filesystem::path temp_file(const char* name) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::error_code ignored;
    std::filesystem::remove(p, ignored);
    return p;
}

}  // namespace

TEST_CASE(nnue_feature_indices) {
    std::set<int> seen;
    for (const Color perspective : {WHITE, BLACK})
        for (const Color c : {WHITE, BLACK})
            for (int pt = PAWN; pt <= KING; ++pt)
                for (int s = 0; s < 64; ++s) {
                    const int f = nnue_feature(perspective, c, PieceType(pt), Square(s));
                    CHECK(f >= 0 && f < kNnueInputs);
                    if (perspective == WHITE) seen.insert(f);
                    // Black sees the board mirrored with the colors swapped.
                    CHECK_EQ(f, nnue_feature(~perspective, ~c, PieceType(pt), Square(s ^ 56)));
                }
    CHECK_EQ(seen.size(), size_t(kNnueInputs));
}

TEST_CASE(nnue_incremental_accumulator) {
    // As tests/test_eval.cpp's check_incremental_updates, with a random network.
    const SearchTables& t = random_tables();
    long long moves_checked = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>& ref_moves) {
            const QuadBoard b = to_board<QuadBoard>(p);
            const SearchBoard<QuadBoard, NnueEval> sb = make_search_board<NnueEval>(b, attacks(), t);
            for (const Move m : ref_moves) {
                SearchBoard<QuadBoard, NnueEval> child = sb;
                const Undo undo = make_search_move(child, m, attacks(), t);
                QuadBoard expected_board = b;
                make_move(expected_board, m);
                const std::string where = move_to_uci(m) + " in " + format_fen(p);
                CHECK_MSG(child.acc == compute_accumulator<NnueEval>(expected_board, t.eval), where);
                unmake_search_move(child, m, undo, sb.key, t);
                CHECK_MSG(child.acc == sb.acc, where + " (unmake)");
                ++moves_checked;
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(moves_checked > 5000);
}

TEST_CASE(nnue_mirror_symmetry_and_float_agreement) {
    const SearchTables& t = random_tables();
    long long nodes = 0;
    double abs_error = 0, max_error = 0, abs_value = 0, max_value = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 8;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            ++nodes;
            const int v = nnue_eval(p, t.eval);
            CHECK_MSG(v == nnue_eval(testutil::mirror(p), t.eval), format_fen(p));
            const double expected = double(random_float_net().forward(make_nnue_sample(p))) * kNnueOutputScale;
            const double err = std::abs(double(v) - expected);
            abs_error += err;
            max_error = std::max(max_error, err);
            abs_value += std::abs(expected);
            max_value = std::max(max_value, std::abs(expected));
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 2, budget, visit);
    }
    REQUIRE(nodes > 500);
    // Rounding (weights to 1/255 and 1/64, the output toward zero) costs a
    // few percent of the output: this random network's scores run to
    // hundreds of centipawns.
    const double mean_error = abs_error / double(nodes), mean_value = abs_value / double(nodes);
    CHECK_MSG(mean_error < 0.03 * mean_value + 2 && max_error < 0.05 * max_value + 10,
              "error mean " + std::to_string(mean_error) + " cp, max " + std::to_string(max_error) +
                  " cp; scores mean " + std::to_string(mean_value) + " cp, max " + std::to_string(max_value) + " cp");
}

// Positions from the test trees, labeled by material balance; every fifth
// one held out.
static NnueDataSet material_data() {
    constexpr int kValue[7] = {0, 100, 300, 300, 500, 900, 0};
    NnueDataSet data;
    long long n = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            const QuadBoard b = to_board<QuadBoard>(p);
            int white = 0;
            for (u64 occ = b.occupied(); occ;) {
                const Square s = Square(pop_lsb(occ));
                white += (b.color(BLACK) & bit(s)) ? -kValue[b.piece_type_on(s)] : kValue[b.piece_type_on(s)];
            }
            NnueSample s = make_nnue_sample(p);
            const int stm = s.stm == 0 ? white : -white;
            s.target = 1.0f / (1.0f + std::pow(10.0f, -float(stm) / kNnueOutputScale));
            (++n % 5 == 0 ? data.validation : data.train).push_back(s);
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    return data;
}

TEST_CASE(nnue_training_learns_material) {
    // The trained network must fit material labels far better than at the
    // start and agree with the labels' sign where material is unequal.
    const NnueDataSet data = material_data();
    REQUIRE(data.train.size() > 2000);
    NnueFloat net = NnueFloat::random(3);
    const double before = nnue_loss(net, data.validation, 4);
    NnueTrainOptions o;
    o.epochs = 30;
    o.batch = 256;
    o.lr = 3e-3f;
    o.threads = 4;
    train_nnue(net, data, o, {});
    const double after = nnue_loss(net, data.validation, 4);
    CHECK_MSG(after < before * 0.3, std::to_string(before) + " -> " + std::to_string(after));
    int agree = 0, unequal = 0;
    for (const NnueSample& s : data.validation) {
        if (std::abs(s.target - 0.5f) < 0.05f) continue;
        ++unequal;
        agree += (net.forward(s) > 0) == (s.target > 0.5f) ? 1 : 0;
    }
    CHECK_MSG(unequal > 100 && agree * 10 >= unequal * 9,
              std::to_string(agree) + " of " + std::to_string(unequal) + " signs right");
}

TEST_CASE(nnue_weight_decay_shrinks_the_weights) {
    // AdamW decay (train-nnue --weight-decay) with a step size so small that
    // the decay dominates: each step multiplies both layers' weights by
    // 1 - lr * decay, and the biases only move by Adam's steps (at most a few
    // times lr each).
    const NnueDataSet data = material_data();
    NnueTrainOptions o;
    o.epochs = 2;  // no step size drop (the last third of 2 epochs is none)
    o.batch = 4096;
    o.lr = 1e-6f;
    o.weight_decay = 1e5f;
    o.threads = 4;
    const NnueFloat start = make_random_float_net(8);
    NnueFloat net = start;
    train_nnue(net, data, o, {});
    const int steps = o.epochs * int((data.train.size() + size_t(o.batch) - 1) / size_t(o.batch));
    const double keep = std::pow(1.0 - double(o.lr) * double(o.weight_decay), steps);
    const double slack = 4.0 * steps * double(o.lr);
    REQUIRE(steps >= 4 && keep < 0.7);
    int bad = 0;
    std::string first;
    const auto expect = [&](const std::vector<float>& now, const std::vector<float>& before, double scale,
                            const char* what) {
        for (size_t i = 0; i < now.size(); ++i) {
            if (std::abs(double(now[i]) - double(before[i]) * scale) <= slack + 1e-6 * std::abs(double(before[i])))
                continue;
            if (bad++ == 0)
                first = std::string(what) + "[" + std::to_string(i) + "]: " + std::to_string(before[i]) + " -> " +
                        std::to_string(now[i]) + ", expected x" + std::to_string(scale);
        }
    };
    expect(net.w1, start.w1, keep, "w1");
    expect(net.w2, start.w2, keep, "w2");
    expect(net.b1, start.b1, 1.0, "b1");
    expect({net.b2}, {start.b2}, 1.0, "b2");
    CHECK_MSG(bad == 0, std::to_string(bad) + " parameters off; first: " + first);
}

TEST_CASE(nnue_files_round_trip) {
    const NnueWeights w = quantize_nnue(random_float_net());
    const std::filesystem::path p = temp_file("chessgpu_test.nnue");
    std::string error;
    REQUIRE_MSG(save_nnue(p.string(), w, error), error);
    NnueWeights back{};
    REQUIRE_MSG(load_nnue(p.string(), back, error), error);
    CHECK(std::memcmp(&w, &back, sizeof w) == 0);
    // A file that isn't one, and a truncated one.
    {
        std::ofstream out(p, std::ios::binary);
        out << "not a network";
    }
    CHECK(!load_nnue(p.string(), back, error));
    REQUIRE_MSG(save_nnue(p.string(), w, error), error);
    std::filesystem::resize_file(p, 100);
    CHECK(!load_nnue(p.string(), back, error));
    std::error_code ignored;
    std::filesystem::remove(p, ignored);
    // The generated header lists the weights in order.
    const std::filesystem::path h = temp_file("chessgpu_test_nnue.cpp");
    REQUIRE_MSG(write_nnue_header(h.string(), w, "test", error), error);
    std::ifstream in(h);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(text.find("const NnueWeights kDefaultNnue") != std::string::npos);
    CHECK(text.find("{" + std::to_string(w.w1[0][0]) + ",") != std::string::npos);
    CHECK(text.find(std::to_string(w.b2) + ",\n};") != std::string::npos);
    in.close();
    std::filesystem::remove(h, ignored);
}

// Central differences in double-checked float arithmetic: a parameter
// passes when the two agree within 1e-4 plus 5%; a few may sit on a kink of
// the clipped ReLU, so 95% must pass. Inputs seen only from black's side get
// a gradient only through black's half of the network.
void check_gradient(NnueFloat net) {
    const int h = net.hidden;
    std::vector<NnueSample> samples;
    std::set<int> white_inputs, black_inputs;
    for (const std::string& fen : testutil::test_fens()) {
        NnueSample s = make_nnue_sample(testutil::parse_or_die(fen));
        s.target = samples.size() % 2 ? 0.8f : 0.25f;
        for (int j = 0; j < s.count; ++j) {
            white_inputs.insert(s.white[j]);
            black_inputs.insert(s.black[j]);
        }
        samples.push_back(s);
    }
    const NnueFloat g = nnue_gradient(net, samples);
    REQUIRE(g.hidden == h && g.w1.size() == net.w1.size() && g.w2.size() == net.w2.size());
    int checked = 0, passed = 0;
    std::string failures;
    const auto check = [&](float& param, float analytic, const std::string& what) {
        const float saved = param;
        const float eps = 1e-3f;
        param = saved + eps;
        const double up = nnue_loss(net, samples, 1);
        param = saved - eps;
        const double down = nnue_loss(net, samples, 1);
        param = saved;
        const double numeric = (up - down) / (2.0 * eps);
        ++checked;
        if (std::abs(numeric - analytic) <= 1e-4 + 0.05 * std::abs(numeric)) {
            ++passed;
        } else if (failures.size() < 400) {
            failures += what + ": " + std::to_string(analytic) + " vs " + std::to_string(numeric) + "; ";
        }
    };
    int black_only = 0;
    for (const int f : black_inputs) {
        const bool only_black = !white_inputs.count(f);
        if (!only_black && f % 7 != 0) continue;  // every black-only input, a sample of the others
        black_only += only_black ? 1 : 0;
        for (int i = 0; i < h; i += 5)
            check(net.w1[size_t(f) * size_t(h) + size_t(i)], g.w1[size_t(f) * size_t(h) + size_t(i)],
                  "w1[" + std::to_string(f) + "][" + std::to_string(i) + "]");
    }
    for (const int f : white_inputs)
        if (f % 5 == 0)
            for (int i = 0; i < h; i += 7)
                check(net.w1[size_t(f) * size_t(h) + size_t(i)], g.w1[size_t(f) * size_t(h) + size_t(i)],
                      "w1[" + std::to_string(f) + "][" + std::to_string(i) + "]");
    for (int i = 0; i < h; ++i) {
        check(net.b1[size_t(i)], g.b1[size_t(i)], "b1[" + std::to_string(i) + "]");
        check(net.w2[size_t(i)], g.w2[size_t(i)], "w2[0][" + std::to_string(i) + "]");
        check(net.w2[size_t(h + i)], g.w2[size_t(h + i)], "w2[1][" + std::to_string(i) + "]");
    }
    check(net.b2, g.b2, "b2");
    REQUIRE(black_only > 10);
    CHECK_MSG(passed * 100 >= checked * 95, std::to_string(h) + " wide: " + std::to_string(passed) + " of " +
                                                std::to_string(checked) + " agree; " + failures);
}

TEST_CASE(nnue_gradient_matches_finite_differences) { check_gradient(random_float_net()); }

// Other widths train too (for comparing losses): the same check at 8 and 48
// neurons per side, and only the search's width quantizes.
TEST_CASE(nnue_other_widths_train) {
    check_gradient(make_random_float_net(8));
    check_gradient(make_random_float_net(48));
    bool threw = false;
    try {
        (void)quantize_nnue(make_random_float_net(48));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)NnueFloat::random(1, kNnueMaxHidden + 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}
