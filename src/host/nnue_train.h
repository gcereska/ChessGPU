#pragma once

// NNUE training (core/nnue.h): fits the float network to
// positions labeled with a search score and the game's result, then
// quantizes it for the search. CPU only, several threads, Adam.
//
// Data lines: "<FEN> | <result> | <game> | <white's score in cp>" (chessgpu
// selfplay writes them; the result is 1, 0.5 or 0 for white). Each position
// trains toward lambda * sigmoid(score / 400) + (1 - lambda) * result, seen
// from the side to move, through sigmoid(network output) (the network's
// output is centipawns / 400, like kNnueOutputScale). Every fifth game is
// held out for validation.

#include <functional>
#include <string>
#include <vector>

#include "core/nnue.h"
#include "core/types.h"
#include "host/position.h"

namespace cb {

// A position as the trainer sees it: the active inputs from each side's
// point of view (at most 32 pieces), the side to move and the target.
struct NnueSample {
    u16 white[32];  // nnue_feature(WHITE, ...) of every piece
    u16 black[32];  // nnue_feature(BLACK, ...)
    u8 count = 0;
    u8 stm = 0;      // 0 white, 1 black
    float target;    // expected score for the side to move, 0..1
};

// The inputs and side to move of p (target 0).
NnueSample make_nnue_sample(const Position& p);

struct NnueDataSet {
    std::vector<NnueSample> train;
    std::vector<NnueSample> validation;
    long long skipped = 0;  // lines without a score, unparsable or in check
};

// Reads data lines (see above); positions whose |score| is at least
// max_score_cp (mates and near-mates) are skipped. lambda weighs the score
// against the result.
NnueDataSet load_nnue_data(const std::string& path, float lambda, int max_score_cp, std::string& error);

// The most hidden neurons per side the trainer takes.
constexpr int kNnueMaxHidden = 512;

// The float network (core/nnue.h's layout, before quantization). Any width
// trains; only kNnueHidden can be quantized for the search (wider ones are
// for comparing losses).
struct NnueFloat {
    int hidden = kNnueHidden;  // neurons per side
    std::vector<float> w1;     // [kNnueInputs][hidden]
    std::vector<float> b1;     // [hidden]
    std::vector<float> w2;     // [2][hidden]
    float b2 = 0;

    // Small random first layer (fixed seed), zero output layer; hidden is
    // 1..kNnueMaxHidden.
    static NnueFloat random(unsigned long long seed, int hidden = kNnueHidden);
    // The output for a sample (centipawns / kNnueOutputScale).
    float forward(const NnueSample& s) const;
};

struct NnueTrainOptions {
    int epochs = 10;
    int batch = 16384;
    float lr = 1e-3f;          // Adam step size, divided by 10 for the last third of the epochs
    // Decoupled weight decay (AdamW) on both layers' weights, not the biases:
    // each step multiplies them by 1 - lr * weight_decay. 0 = off.
    float weight_decay = 0;
    int threads = 8;
    unsigned long long seed = 1;
};

struct NnueEpochStats {
    int epoch = 0;
    double train_loss = 0;
    double validation_loss = 0;
    double seconds = 0;
};

// Trains net in place; `progress` (may be empty) is called after each epoch.
void train_nnue(NnueFloat& net, const NnueDataSet& data, const NnueTrainOptions& options,
                const std::function<void(const NnueEpochStats&)>& progress);

// Mean squared error of sigmoid(output) against the targets.
double nnue_loss(const NnueFloat& net, const std::vector<NnueSample>& samples, int threads);

// The gradient of nnue_loss with respect to every weight, in NnueFloat's
// layout, computed by the training's backpropagation (tests compare it
// with finite differences).
NnueFloat nnue_gradient(const NnueFloat& net, const std::vector<NnueSample>& samples);

// Rounds the float network to core/nnue.h's integers, clamping the first
// layer so no accumulator of a legal position can overflow 16 bits. The
// network must have kNnueHidden neurons per side (else invalid_argument).
NnueWeights quantize_nnue(const NnueFloat& net);

// Weights files: a header (the tag "CBNN", a version and the layer sizes), then the
// quantized weights, little-endian.
bool save_nnue(const std::string& path, const NnueWeights& w, std::string& error);
bool load_nnue(const std::string& path, NnueWeights& w, std::string& error);

// The weights as a C++ header defining kDefaultNnue (core/nnue_default.h).
bool write_nnue_header(const std::string& path, const NnueWeights& w, const std::string& comment, std::string& error);

}  // namespace cb
