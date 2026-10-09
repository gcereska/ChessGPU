// Texel tuning (host/tune.h):
// - a position's base and features reproduce the terms evaluation (white's
//   point of view, before the engine's integer rounding);
// - the gradient matches finite differences of the loss;
// - on results generated from known weights, the fit recovers them;
// - the L2 pull keeps weights at the anchor when it dominates.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "host/fen.h"
#include "host/host_tables.h"
#include "host/tune.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

double eval_with(const TuningPosition& p, const TuneWeights& w) {
    double e = p.base;
    for (int i = 0; i < kTuneWeights; ++i) e += w.w[i] * p.features[i];
    return e;
}

TuningPositions sample_positions() {
    TuningPositions v;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = 300;
        auto visit = [&](const Position& p, const std::vector<Move>&) { v.push_back(tuning_position(p, 0.5f)); };
        testutil::walk_tree(testutil::parse_or_die(fen), 2, budget, visit);
    }
    return v;
}

}  // namespace

TEST_CASE(tune_features_reproduce_the_eval) {
    const TuneWeights w = current_term_weights();
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const QuadBoard b = to_board<QuadBoard>(p);
        const int engine = evaluate_position<PestoTermsEval>(
            b, compute_accumulator<PestoTermsEval>(b, host_search_tables().eval), WHITE);
        const double tuner = eval_with(tuning_position(p, 0.5f), w);
        CHECK_MSG(std::fabs(tuner - engine) < 1.0, fen + ": " + std::to_string(tuner) + " vs " + std::to_string(engine));
    }
}

TEST_CASE(tune_gradient_matches_finite_differences) {
    TuningPositions set = sample_positions();
    std::mt19937 rng(7);
    for (TuningPosition& p : set) p.result = float(rng() % 3) / 2.0f;
    const TuneWeights w = current_term_weights();
    const double k = 1.2;
    TuneWeights g;
    tuning_gradient(set, w, k, g);
    for (int i = 0; i < kTuneWeights; ++i) {
        TuneWeights up = w, down = w;
        up.w[i] += 0.5;
        down.w[i] -= 0.5;
        const double numeric = (tuning_loss(set, up, k) - tuning_loss(set, down, k)) / 1.0;
        CHECK_MSG(std::fabs(numeric - g.w[i]) <= 1e-6 + 1e-3 * std::fabs(numeric),
                  "weight " + std::to_string(i) + ": " + std::to_string(g.w[i]) + " vs " + std::to_string(numeric));
    }
}

TEST_CASE(tune_recovers_known_weights) {
    // Results are the logistic curve of an evaluation with known weights;
    // fitting from the engine's weights must find them again.
    TuningPositions set = sample_positions();
    TuneWeights truth = current_term_weights();
    truth.w[kTermBishopPair] += 20;
    truth.w[kTermCount + kTermRookMobility] -= 3;
    truth.w[kTermDoubledPawn] -= 15;
    const double k = 1.3;
    for (TuningPosition& p : set) p.result = float(1.0 / (1.0 + std::pow(10.0, -k * eval_with(p, truth) / 400.0)));
    TuneOptions options;
    options.iterations = 4000;
    const TuneWeights fit = tune_weights(set, k, current_term_weights(), options);
    CHECK_MSG(tuning_loss(set, fit, k) < 1e-6, std::to_string(tuning_loss(set, fit, k)));
    for (const int i : {int(kTermBishopPair), kTermCount + int(kTermRookMobility), int(kTermDoubledPawn)})
        CHECK_MSG(std::fabs(fit.w[i] - truth.w[i]) < 2.0,
                  "weight " + std::to_string(i) + ": " + std::to_string(fit.w[i]) + " vs " + std::to_string(truth.w[i]));
}

TEST_CASE(tune_l2_pull_keeps_the_anchor) {
    TuningPositions set = sample_positions();
    std::mt19937 rng(3);
    for (TuningPosition& p : set) p.result = float(rng() % 3) / 2.0f;
    TuneOptions options;
    options.iterations = 500;
    options.anchor = current_term_weights();
    options.l2 = 10.0;  // dominates the loss
    const TuneWeights fit = tune_weights(set, 1.2, current_term_weights(), options);
    for (int i = 0; i < kTuneWeights; ++i) CHECK(std::fabs(fit.w[i] - options.anchor.w[i]) < 0.5);
    CHECK(format_term_weights(current_term_weights()).find("case kTermBishopPair: return endgame ? 50 : 30;") !=
          std::string::npos);
}
