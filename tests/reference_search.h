#pragma once

// Deliberately simple alpha-beta search over the mailbox rules in
// reference_chess.h, sharing no search, evaluation or move generation code
// with the engine (only the PeSTO numbers). The root score of a full-window
// search is the exact minimax value whatever the move order or pruning, so
// it must equal the engine's score exactly. Slow; for small depths.
//
// Same rules as core/search.h: mate scores by ply, stalemate / 50-move rule /
// insufficient material / repetition are draws, positions at ply 64 return
// their static evaluation, quiescence tries captures (promoting ones only to
// a queen) and pushes promoting to a queen, or every move when in check.

#include <algorithm>
#include <array>
#include <vector>

#include "core/pesto_tables.h"
#include "reference_chess.h"

namespace refsearch {

constexpr int kMate = 32000;
constexpr int kInfinite = 32001;
constexpr int kMaxPly = 64;

enum class Eval { Material, Pesto };

inline int evaluate(const ref::Board& b, Eval eval) {
    static constexpr int kValue[7] = {0, 100, 320, 330, 500, 900, 0};
    int mg = 0;
    int eg = 0;
    int phase = 0;
    for (int sq = 0; sq < 64; ++sq) {
        const int code = b.sq[sq];
        if (!code) continue;
        const int type = ref::type_of(code);
        const bool white = ref::is_color(code, cb::WHITE);
        const int sign = white ? 1 : -1;
        if (eval == Eval::Material) {
            mg += sign * kValue[type];
            continue;
        }
        // The tables list rank 8 first, as seen by white; black reads them
        // as if the board were turned around.
        const int rank = sq / 8;
        const int file = sq % 8;
        const int row = white ? 7 - rank : rank;
        const int index = row * 8 + file;
        mg += sign * (cb::pesto::kMgValue[type - 1] + cb::pesto::kMgTables[type - 1][index]);
        eg += sign * (cb::pesto::kEgValue[type - 1] + cb::pesto::kEgTables[type - 1][index]);
        phase += cb::pesto::kPhaseInc[type - 1];
    }
    int white_score = mg;
    if (eval == Eval::Pesto) {
        const int mg_phase = std::min(phase, 24);
        white_score = (mg * mg_phase + eg * (24 - mg_phase)) / 24;
    }
    return b.side == cb::WHITE ? white_score : -white_score;
}

inline bool in_check(const ref::Board& b) { return ref::attacked(b, ref::find_king(b, b.side), ~b.side); }

// What the repetition rule compares: placement, side, castling rights and the
// en passant square only if an en passant capture is legal.
struct Identity {
    std::array<int, 64> sq;
    int side;
    unsigned castling;
    int ep;
    bool operator==(const Identity&) const = default;
};

inline Identity identity_of(const ref::Board& b) {
    Identity id;
    for (int i = 0; i < 64; ++i) id.sq[i] = b.sq[i];
    id.side = int(b.side);
    id.castling = b.castling;
    id.ep = -1;
    for (const cb::Move m : ref::legal_moves(b))
        if (m.flags() == cb::EP_CAPTURE) id.ep = b.ep;
    return id;
}

inline bool insufficient(const ref::Board& b) {
    int minors = 0;
    for (int i = 0; i < 64; ++i) {
        const int t = b.sq[i] ? ref::type_of(b.sq[i]) : 0;
        if (t == cb::PAWN || t == cb::ROOK || t == cb::QUEEN) return false;
        if (t == cb::KNIGHT || t == cb::BISHOP) ++minors;
    }
    return minors <= 1;
}

struct Searcher {
    Eval eval;
    // Positions from the start of the game up to the current node:
    // path[game_length] is the search root.
    std::vector<Identity> path;
    std::vector<int> halfmoves;
    std::vector<bool> repeated;  // for game positions and the root: occurred earlier
    int game_length = 0;
    long long nodes = 0;

    // Does position i occur earlier within its halfmove clock (from 4 plies back)?
    bool repeats_earlier(int i) const {
        for (int d = 4; d <= halfmoves[i] && d <= i; d += 2)
            if (path[i - d] == path[i]) return true;
        return false;
    }

    bool is_repetition(int ply) const {
        const int i = game_length + ply;
        for (int d = 4; d <= halfmoves[i] && d <= i; d += 2) {
            if (path[i - d] == path[i]) {
                const int earlier_ply = ply - d;
                return earlier_ply > 0 || repeated[i - d];
            }
        }
        return false;
    }

    int search(const ref::Board& b, int depth, int alpha, int beta, int ply) {
        ++nodes;
        path.resize(size_t(game_length + ply + 1));
        halfmoves.resize(path.size());
        path[size_t(game_length + ply)] = identity_of(b);
        halfmoves[size_t(game_length + ply)] = b.halfmove;

        const std::vector<cb::Move> legal = ref::legal_moves(b);
        const bool check = in_check(b);
        if (ply > 0) {
            if (is_repetition(ply) || insufficient(b)) return 0;
            if (b.halfmove >= 100) return (check && legal.empty()) ? -kMate + ply : 0;
        }
        if (ply >= kMaxPly) return evaluate(b, eval);

        std::vector<cb::Move> moves;
        if (depth <= 0 && !check) {
            const int stand = evaluate(b, eval);
            if (stand >= beta) return beta;  // fail-hard
            alpha = std::max(alpha, stand);
            for (const cb::Move m : legal) {
                const bool capture = b.sq[m.to()] != 0 || m.flags() == cb::EP_CAPTURE;
                const bool queen_promotion = m.is_promotion() && m.promotion_type() == cb::QUEEN;
                if ((capture && (!m.is_promotion() || queen_promotion)) || (!capture && queen_promotion))
                    moves.push_back(m);
            }
        } else {
            if (legal.empty()) return check ? -kMate + ply : 0;
            moves = legal;
        }
        // Captures of valuable pieces first, only to prune more.
        std::stable_sort(moves.begin(), moves.end(), [&](cb::Move x, cb::Move y) {
            return ref::type_of(b.sq[x.to()]) > ref::type_of(b.sq[y.to()]);
        });
        for (const cb::Move m : moves) {
            const int score = -search(ref::play(b, m), depth - 1, -beta, -alpha, ply + 1);
            if (score >= beta) return beta;
            alpha = std::max(alpha, score);
        }
        return alpha;
    }
};

// Exact score of `root` searched to `depth`, given the game positions before
// it (oldest first; may be empty).
inline int search_root(const ref::Board& root, const std::vector<ref::Board>& game, int depth, Eval eval,
                       long long* nodes = nullptr) {
    Searcher s;
    s.eval = eval;
    s.game_length = int(game.size());
    for (const ref::Board& g : game) {
        s.path.push_back(identity_of(g));
        s.halfmoves.push_back(g.halfmove);
    }
    s.path.push_back(identity_of(root));
    s.halfmoves.push_back(root.halfmove);
    s.repeated.assign(s.path.size(), false);
    for (int i = 0; i <= s.game_length; ++i) s.repeated[size_t(i)] = s.repeats_earlier(i);
    const int score = s.search(root, depth, -kInfinite, kInfinite, 0);
    if (nodes) *nodes = s.nodes;
    return score;
}

}  // namespace refsearch
