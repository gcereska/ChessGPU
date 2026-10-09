#include "host/tune.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "core/see.h"
#include "host/fen.h"
#include "host/host_tables.h"

namespace cb {

namespace {

constexpr const char* kTermNames[kTermCount] = {
    "kTermBishopPair",     "kTermDoubledPawn",    "kTermIsolatedPawn",   "kTermPassedPawn2",
    "kTermPassedPawn3",    "kTermPassedPawn4",    "kTermPassedPawn5",    "kTermPassedPawn6",
    "kTermPassedPawn7",    "kTermKnightMobility", "kTermBishopMobility", "kTermRookMobility",
    "kTermQueenMobility",  "kTermRookOpenFile",   "kTermRookSemiOpenFile", "kTermKingShield1",
    "kTermKingShield2",
};

double sigmoid(double eval, double k) { return 1.0 / (1.0 + std::pow(10.0, -k * eval / 400.0)); }

double eval_of(const TuningPosition& p, const TuneWeights& w) {
    double e = p.base;
    for (int i = 0; i < kTuneWeights; ++i) e += w.w[i] * p.features[i];
    return e;
}

// Quiet: not in check, and no capture or promotion wins material by SEE.
bool is_quiet(const QuadBoard& b) {
    const HqAttacks a(host_attack_tables());
    if (side_to_move_in_check(b, a)) return false;
    MoveList list;
    generate_legal(b, a, list);
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        if ((m.is_capture() || m.is_promotion()) && see(b, a, m) > 0) return false;
    }
    return true;
}

}  // namespace

TuningPosition tuning_position(const Position& p, float result) {
    const QuadBoard b = to_board<QuadBoard>(p);
    const PestoEval::Accumulator acc = compute_accumulator<PestoEval>(b, host_search_tables().eval);
    const int mg_phase = acc.phase < pesto::kPhaseMax ? acc.phase : pesto::kPhaseMax;
    const int eg_phase = pesto::kPhaseMax - mg_phase;
    TuningPosition t{};
    t.base = float(double(unpack_mg(acc.packed)) * mg_phase / pesto::kPhaseMax +
                   double(unpack_eg(acc.packed)) * eg_phase / pesto::kPhaseMax);
    const TermCounts c = count_terms(b);
    for (int i = 0; i < kTermCount; ++i) {
        t.features[i] = float(double(c.v[i]) * mg_phase / pesto::kPhaseMax);
        t.features[kTermCount + i] = float(double(c.v[i]) * eg_phase / pesto::kPhaseMax);
    }
    t.result = result;
    return t;
}

TuningSet load_tuning_set(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("can't read " + path);
    TuningSet set;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t bar = line.find('|');
        if (bar == std::string::npos) throw std::runtime_error("no result in line: " + line);
        const size_t bar2 = line.find('|', bar + 1);
        Position p;
        std::string error;
        if (!parse_fen(line.substr(0, bar), p, error)) throw std::runtime_error(error + ": " + line);
        const size_t result_length = bar2 == std::string::npos ? std::string::npos : bar2 - bar - 1;
        const float result = std::stof(line.substr(bar + 1, result_length));
        const long game = bar2 == std::string::npos ? -1 : std::stol(line.substr(bar2 + 1));
        ++set.lines;
        if (!is_quiet(to_board<QuadBoard>(p))) {
            ++set.rejected;
            continue;
        }
        (game >= 0 && game % 5 == 0 ? set.validation : set.training).push_back(tuning_position(p, result));
    }
    return set;
}

TuneWeights current_term_weights() {
    TuneWeights w{};
    for (int i = 0; i < kTermCount; ++i) {
        w.w[i] = term_weight(i, false);
        w.w[kTermCount + i] = term_weight(i, true);
    }
    return w;
}

double tuning_loss(const TuningPositions& set, const TuneWeights& weights, double k) {
    double sum = 0;
    for (const TuningPosition& p : set) {
        const double d = p.result - sigmoid(eval_of(p, weights), k);
        sum += d * d;
    }
    return set.empty() ? 0 : sum / double(set.size());
}

void tuning_gradient(const TuningPositions& set, const TuneWeights& weights, double k, TuneWeights& out) {
    out = TuneWeights{};
    const double scale = k * std::log(10.0) / 400.0;
    for (const TuningPosition& p : set) {
        const double s = sigmoid(eval_of(p, weights), k);
        // d/dw (r - s)^2 = -2 (r - s) s (1 - s) scale * feature
        const double g = -2.0 * (p.result - s) * s * (1.0 - s) * scale;
        for (int i = 0; i < kTuneWeights; ++i) out.w[i] += g * p.features[i];
    }
    const double n = set.empty() ? 1.0 : double(set.size());
    for (double& x : out.w) x /= n;
}

double fit_scale(const TuningPositions& set, const TuneWeights& weights) {
    double lo = 0.1, hi = 3.0;
    const double phi = (std::sqrt(5.0) - 1) / 2;
    double a = hi - phi * (hi - lo), b = lo + phi * (hi - lo);
    double fa = tuning_loss(set, weights, a), fb = tuning_loss(set, weights, b);
    for (int i = 0; i < 40; ++i) {
        if (fa < fb) {
            hi = b;
            b = a;
            fb = fa;
            a = hi - phi * (hi - lo);
            fa = tuning_loss(set, weights, a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + phi * (hi - lo);
            fb = tuning_loss(set, weights, b);
        }
    }
    return (lo + hi) / 2;
}

TuneWeights tune_weights(const TuningPositions& set, double k, const TuneWeights& start, const TuneOptions& options) {
    TuneWeights w = start, m{}, v{}, g{};
    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-12;
    for (int t = 1; t <= options.iterations; ++t) {
        tuning_gradient(set, w, k, g);
        for (int i = 0; i < kTuneWeights; ++i) g.w[i] += 2 * options.l2 * (w.w[i] - options.anchor.w[i]);
        const double c1 = 1 - std::pow(beta1, t), c2 = 1 - std::pow(beta2, t);
        for (int i = 0; i < kTuneWeights; ++i) {
            m.w[i] = beta1 * m.w[i] + (1 - beta1) * g.w[i];
            v.w[i] = beta2 * v.w[i] + (1 - beta2) * g.w[i] * g.w[i];
            w.w[i] -= options.learning_rate * (m.w[i] / c1) / (std::sqrt(v.w[i] / c2) + eps);
        }
    }
    return w;
}

std::string format_term_weights(const TuneWeights& weights) {
    std::string s;
    char line[160];
    for (int i = 0; i < kTermCount; ++i) {
        std::snprintf(line, sizeof line, "        case %s: return endgame ? %ld : %ld;\n", kTermNames[i],
                      std::lround(weights.w[kTermCount + i]), std::lround(weights.w[i]));
        s += line;
    }
    return s;
}

}  // namespace cb
