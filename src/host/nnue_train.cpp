#include "host/nnue_train.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "core/movegen.h"
#include "host/fen.h"
#include "host/host_tables.h"

namespace cb {

namespace {

constexpr float kLn10 = 2.302585093f;
// |w1| at most this, so 32 pieces' worth of weights fit an int16 at scale kNnueQA.
constexpr float kMaxW1 = 32767.0f / (32.0f * kNnueQA) - 0.01f;
constexpr float kMaxB1 = 32767.0f / kNnueQA - 1.0f;
constexpr float kMaxW2 = 32767.0f / kNnueQB - 1.0f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Expected score of a position worth cp centipawns (logistic, base 10, 400 cp per decade).
float expectation(float cp) { return sigmoid(cp * kLn10 / kNnueOutputScale); }

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream in(line);
    while (std::getline(in, cur, sep)) {
        const size_t a = cur.find_first_not_of(' ');
        const size_t b = cur.find_last_not_of(' ');
        parts.push_back(a == std::string::npos ? std::string() : cur.substr(a, b - a + 1));
    }
    return parts;
}

// Per-thread gradients and activations.
struct Grad {
    std::vector<float> w1, b1, w2;
    double b2 = 0;
    double loss = 0;

    explicit Grad(int hidden) : w1(size_t(kNnueInputs) * size_t(hidden)), b1(size_t(hidden)), w2(2 * size_t(hidden)) {}
    void clear() {
        std::fill(w1.begin(), w1.end(), 0.0f);
        std::fill(b1.begin(), b1.end(), 0.0f);
        std::fill(w2.begin(), w2.end(), 0.0f);
        b2 = 0;
        loss = 0;
    }
};

// First-layer sums of a sample from each side's point of view (biases included).
void accumulate(const NnueFloat& n, const NnueSample& s, float* white, float* black) {
    const int h = n.hidden;
    for (int i = 0; i < h; ++i) white[i] = black[i] = n.b1[size_t(i)];
    for (int j = 0; j < s.count; ++j) {
        const float* w = &n.w1[size_t(s.white[j]) * size_t(h)];
        const float* b = &n.w1[size_t(s.black[j]) * size_t(h)];
        for (int i = 0; i < h; ++i) {
            white[i] += w[i];
            black[i] += b[i];
        }
    }
}

float clip01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Adds the sample's loss gradient to g.
void backprop(const NnueFloat& n, const NnueSample& s, Grad& g) {
    const int h = n.hidden;
    float white[kNnueMaxHidden], black[kNnueMaxHidden];
    accumulate(n, s, white, black);
    const float* us = s.stm == 0 ? white : black;
    const float* them = s.stm == 0 ? black : white;
    float y = n.b2;
    for (int i = 0; i < h; ++i) y += n.w2[size_t(i)] * clip01(us[i]) + n.w2[size_t(h + i)] * clip01(them[i]);
    const float p = sigmoid(y * kLn10);
    const float err = p - s.target;
    g.loss += double(err) * err;
    const float dy = 2.0f * err * p * (1.0f - p) * kLn10;
    g.b2 += dy;
    float d_us[kNnueMaxHidden], d_them[kNnueMaxHidden];
    for (int i = 0; i < h; ++i) {
        g.w2[size_t(i)] += dy * clip01(us[i]);
        g.w2[size_t(h + i)] += dy * clip01(them[i]);
        d_us[i] = (us[i] > 0 && us[i] < 1) ? dy * n.w2[size_t(i)] : 0.0f;
        d_them[i] = (them[i] > 0 && them[i] < 1) ? dy * n.w2[size_t(h + i)] : 0.0f;
    }
    const float* d_white = s.stm == 0 ? d_us : d_them;
    const float* d_black = s.stm == 0 ? d_them : d_us;
    for (int i = 0; i < h; ++i) g.b1[size_t(i)] += d_white[i] + d_black[i];
    for (int j = 0; j < s.count; ++j) {
        float* w = &g.w1[size_t(s.white[j]) * size_t(h)];
        float* b = &g.w1[size_t(s.black[j]) * size_t(h)];
        for (int i = 0; i < h; ++i) {
            w[i] += d_white[i];
            b[i] += d_black[i];
        }
    }
}

struct Adam {
    std::vector<float> m, v;
    explicit Adam(size_t n) : m(n, 0.0f), v(n, 0.0f) {}
    // decay: decoupled weight decay (AdamW), applied before the update.
    void step(float* param, const float* grad, size_t n, float lr, int t, float clamp, float decay = 0) {
        constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        const float c1 = 1.0f - std::pow(b1, float(t));
        const float c2 = 1.0f - std::pow(b2, float(t));
        const float keep = 1.0f - lr * decay;
        for (size_t i = 0; i < n; ++i) {
            m[i] = b1 * m[i] + (1 - b1) * grad[i];
            v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
            if (decay != 0) param[i] *= keep;
            param[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
            param[i] = std::clamp(param[i], -clamp, clamp);
        }
    }
};

template <class F>
void parallel_for(int threads, size_t n, F&& f) {
    std::vector<std::thread> pool;
    const size_t chunk = (n + size_t(threads) - 1) / size_t(threads);
    for (int t = 0; t < threads; ++t) {
        const size_t begin = size_t(t) * chunk, end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&, t, begin, end] { f(t, begin, end); });
    }
    for (std::thread& th : pool) th.join();
}

}  // namespace

NnueSample make_nnue_sample(const Position& p) {
    const QuadBoard b = to_board<QuadBoard>(p);
    NnueSample s;
    s.target = 0;
    const u64 black_pieces = b.color(BLACK);
    for (u64 occ = b.occupied(); occ && s.count < 32;) {
        const Square sq = Square(pop_lsb(occ));
        const Color c = (black_pieces & bit(sq)) ? BLACK : WHITE;
        const PieceType pt = b.piece_type_on(sq);
        s.white[s.count] = u16(nnue_feature(WHITE, c, pt, sq));
        s.black[s.count] = u16(nnue_feature(BLACK, c, pt, sq));
        ++s.count;
    }
    s.stm = state::side(p.state) == WHITE ? 0 : 1;
    return s;
}

NnueDataSet load_nnue_data(const std::string& path, float lambda, int max_score_cp, std::string& error) {
    NnueDataSet data;
    std::ifstream in(path);
    if (!in) {
        error = "can't read " + path;
        return data;
    }
    const HqAttacks a(host_attack_tables());
    for (std::string line; std::getline(in, line);) {
        const std::vector<std::string> f = split(line, '|');
        if (f.size() < 4 || f[3].empty()) {
            ++data.skipped;
            continue;
        }
        Position p;
        std::string fen_error;
        if (!parse_fen(f[0], p, fen_error)) {
            ++data.skipped;
            continue;
        }
        const float result = std::stof(f[1]);
        const long long game = std::stoll(f[2]);
        const int score = std::stoi(f[3]);
        const QuadBoard b = to_board<QuadBoard>(p);
        if (std::abs(score) >= max_score_cp || side_to_move_in_check(b, a)) {
            ++data.skipped;
            continue;
        }
        NnueSample s = make_nnue_sample(p);
        const bool white_to_move = s.stm == 0;
        const float score_stm = float(white_to_move ? score : -score);
        const float result_stm = white_to_move ? result : 1.0f - result;
        s.target = lambda * expectation(score_stm) + (1.0f - lambda) * result_stm;
        (game % 5 == 0 ? data.validation : data.train).push_back(s);
    }
    return data;
}

NnueFloat NnueFloat::random(unsigned long long seed, int hidden) {
    if (hidden < 1 || hidden > kNnueMaxHidden) throw std::invalid_argument("NNUE width out of range");
    NnueFloat n;
    n.hidden = hidden;
    n.w1.resize(size_t(kNnueInputs) * size_t(hidden));
    n.b1.assign(size_t(hidden), 0.0f);
    n.w2.assign(2 * size_t(hidden), 0.0f);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (float& w : n.w1) w = dist(rng);
    for (float& b : n.b1) b = 0.5f;  // start in the linear part of the clipped ReLU
    return n;
}

float NnueFloat::forward(const NnueSample& s) const {
    float white[kNnueMaxHidden], black[kNnueMaxHidden];
    accumulate(*this, s, white, black);
    const float* us = s.stm == 0 ? white : black;
    const float* them = s.stm == 0 ? black : white;
    float y = b2;
    for (int i = 0; i < hidden; ++i) y += w2[size_t(i)] * clip01(us[i]) + w2[size_t(hidden + i)] * clip01(them[i]);
    return y;
}

double nnue_loss(const NnueFloat& net, const std::vector<NnueSample>& samples, int threads) {
    if (samples.empty()) return 0;
    std::vector<double> sums(size_t(std::max(1, threads)), 0.0);
    parallel_for(std::max(1, threads), samples.size(), [&](int t, size_t begin, size_t end) {
        double s = 0;
        for (size_t i = begin; i < end; ++i) {
            const float err = sigmoid(net.forward(samples[i]) * kLn10) - samples[i].target;
            s += double(err) * err;
        }
        sums[size_t(t)] = s;
    });
    return std::accumulate(sums.begin(), sums.end(), 0.0) / double(samples.size());
}

NnueFloat nnue_gradient(const NnueFloat& net, const std::vector<NnueSample>& samples) {
    Grad g(net.hidden);
    g.clear();
    for (const NnueSample& s : samples) backprop(net, s, g);
    const float scale = samples.empty() ? 0.0f : 1.0f / float(samples.size());
    NnueFloat out;
    out.hidden = net.hidden;
    out.w1 = g.w1;
    out.b1 = g.b1;
    out.w2 = g.w2;
    for (float& x : out.w1) x *= scale;
    for (float& x : out.b1) x *= scale;
    for (float& x : out.w2) x *= scale;
    out.b2 = float(g.b2) * scale;
    return out;
}

void train_nnue(NnueFloat& net, const NnueDataSet& data, const NnueTrainOptions& o,
                const std::function<void(const NnueEpochStats&)>& progress) {
    const int threads = std::max(1, o.threads);
    std::vector<Grad> grads(static_cast<size_t>(threads), Grad(net.hidden));
    Grad total(net.hidden);
    Adam adam_w1(net.w1.size()), adam_b1(net.b1.size()), adam_w2(net.w2.size()), adam_b2(1);
    std::vector<size_t> order(data.train.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::mt19937_64 rng(o.seed);
    int step = 0;
    for (int epoch = 1; epoch <= o.epochs; ++epoch) {
        const auto start = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);
        const float lr = epoch > o.epochs - o.epochs / 3 ? o.lr / 10 : o.lr;
        double epoch_loss = 0;
        for (size_t first = 0; first < order.size(); first += size_t(o.batch)) {
            const size_t last = std::min(order.size(), first + size_t(o.batch));
            parallel_for(threads, last - first, [&](int t, size_t begin, size_t end) {
                Grad& g = grads[size_t(t)];
                g.clear();
                for (size_t i = begin; i < end; ++i) backprop(net, data.train[order[first + i]], g);
            });
            total.clear();
            for (const Grad& g : grads) {
                for (size_t i = 0; i < total.w1.size(); ++i) total.w1[i] += g.w1[i];
                for (size_t i = 0; i < total.b1.size(); ++i) total.b1[i] += g.b1[i];
                for (size_t i = 0; i < total.w2.size(); ++i) total.w2[i] += g.w2[i];
                total.b2 += g.b2;
                total.loss += g.loss;
            }
            const float scale = 1.0f / float(last - first);
            for (float& x : total.w1) x *= scale;
            for (float& x : total.b1) x *= scale;
            for (float& x : total.w2) x *= scale;
            const float gb2 = float(total.b2) * scale;
            ++step;
            adam_w1.step(net.w1.data(), total.w1.data(), net.w1.size(), lr, step, kMaxW1, o.weight_decay);
            adam_b1.step(net.b1.data(), total.b1.data(), net.b1.size(), lr, step, kMaxB1);
            adam_w2.step(net.w2.data(), total.w2.data(), net.w2.size(), lr, step, kMaxW2, o.weight_decay);
            adam_b2.step(&net.b2, &gb2, 1, lr, step, 1e6f);
            epoch_loss += total.loss;
        }
        NnueEpochStats st;
        st.epoch = epoch;
        st.train_loss = order.empty() ? 0 : epoch_loss / double(order.size());
        st.validation_loss = nnue_loss(net, data.validation, threads);
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (progress) progress(st);
    }
}

NnueWeights quantize_nnue(const NnueFloat& n) {
    if (n.hidden != kNnueHidden) throw std::invalid_argument("only 768x32 networks quantize for this build");
    NnueWeights w{};
    const auto q = [](float x, float scale, float max_abs) {
        return std::lround(std::clamp(x, -max_abs, max_abs) * scale);
    };
    for (int f = 0; f < kNnueInputs; ++f)
        for (int i = 0; i < kNnueHidden; ++i)
            w.w1[f][i] = i16(q(n.w1[size_t(f) * kNnueHidden + size_t(i)], kNnueQA, kMaxW1));
    for (int i = 0; i < kNnueHidden; ++i) {
        w.b1[i] = i16(q(n.b1[size_t(i)], kNnueQA, kMaxB1));
        w.w2[0][i] = i16(q(n.w2[size_t(i)], kNnueQB, kMaxW2));
        w.w2[1][i] = i16(q(n.w2[size_t(kNnueHidden + i)], kNnueQB, kMaxW2));
    }
    w.b2 = i32(std::llround(double(n.b2) * kNnueQA * kNnueQB));
    return w;
}

namespace {

constexpr char kMagic[4] = {'C', 'B', 'N', 'N'};
constexpr u32 kVersion = 1;

}  // namespace

bool save_nnue(const std::string& path, const NnueWeights& w, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "can't write " + path;
        return false;
    }
    const u32 header[3] = {kVersion, u32(kNnueInputs), u32(kNnueHidden)};
    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(header), sizeof header);
    out.write(reinterpret_cast<const char*>(&w), sizeof w);
    if (!out) {
        error = "write failed: " + path;
        return false;
    }
    return true;
}

bool load_nnue(const std::string& path, NnueWeights& w, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "can't read " + path;
        return false;
    }
    char magic[4];
    u32 header[3];
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(header), sizeof header);
    if (!in || std::memcmp(magic, kMagic, 4) != 0 || header[0] != kVersion) {
        error = path + " is not a ChessGPU NNUE file (version 1)";
        return false;
    }
    if (header[1] != u32(kNnueInputs) || header[2] != u32(kNnueHidden)) {
        error = path + ": the network is " + std::to_string(header[1]) + "x" + std::to_string(header[2]) +
                ", this build reads " + std::to_string(kNnueInputs) + "x" + std::to_string(kNnueHidden);
        return false;
    }
    in.read(reinterpret_cast<char*>(&w), sizeof w);
    if (!in) {
        error = path + " is truncated";
        return false;
    }
    return true;
}

bool write_nnue_header(const std::string& path, const NnueWeights& w, const std::string& comment, std::string& error) {
    std::ofstream out(path);
    if (!out) {
        error = "can't write " + path;
        return false;
    }
    out << "// Generated by chessgpu train-nnue (host/nnue_train.cpp): " << comment << "\n"
        << "// The engine's default NNUE weights (core/nnue.h).\n\n"
        << "#include \"core/nnue.h\"\n\nnamespace cb {\n\nconst NnueWeights kDefaultNnue = {\n    {\n";
    for (int f = 0; f < kNnueInputs; ++f) {
        out << "        {";
        for (int i = 0; i < kNnueHidden; ++i) out << w.w1[f][i] << (i + 1 < kNnueHidden ? "," : "");
        out << "},\n";
    }
    out << "    },\n    {";
    for (int i = 0; i < kNnueHidden; ++i) out << w.b1[i] << (i + 1 < kNnueHidden ? "," : "");
    out << "},\n    {\n";
    for (int h = 0; h < 2; ++h) {
        out << "        {";
        for (int i = 0; i < kNnueHidden; ++i) out << w.w2[h][i] << (i + 1 < kNnueHidden ? "," : "");
        out << "},\n";
    }
    out << "    },\n    " << w.b2 << ",\n};\n\n}  // namespace cb\n";
    if (!out) {
        error = "write failed: " + path;
        return false;
    }
    return true;
}

}  // namespace cb
