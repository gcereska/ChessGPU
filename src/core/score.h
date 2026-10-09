#pragma once

// Search scores, in centipawns from the side to move's point of view.

#include "core/types.h"

namespace cb {

// Deepest ply (distance from the search root) any search reaches, counting
// quiescence. A node at this ply is not searched further: it returns its
// static evaluation.
constexpr int kMaxPly = 64;

constexpr int kScoreDraw = 0;
// Mate found `ply` plies from the root scores kScoreMate - ply, so shorter
// mates score higher. Being mated scores the negation.
constexpr int kScoreMate = 32000;
// Every score outside [-kScoreMateBound, kScoreMateBound] is a mate score.
constexpr int kScoreMateBound = kScoreMate - kMaxPly;
// Bounds for a full window; never returned as a score.
constexpr int kScoreInfinite = kScoreMate + 1;

CB_HD constexpr int mate_in(int ply) { return kScoreMate - ply; }
CB_HD constexpr int mated_in(int ply) { return -kScoreMate + ply; }
CB_HD constexpr bool is_mate_score(int score) { return score > kScoreMateBound || score < -kScoreMateBound; }

}  // namespace cb
