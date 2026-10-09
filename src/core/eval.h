#pragma once

// Evaluation policies. Each one keeps an accumulator that make/unmake update
// incrementally (add/remove a piece on a square), and turns it into a score
// for the side to move at the leaves. Policies are template parameters of the
// search, like the board, so each kernel is compiled for one of them.
//
//   Policy::Accumulator                         running state, a few registers
//   Policy::add(acc, tables, c, pt, s)          piece (c, pt) appears on s
//   Policy::remove(acc, tables, c, pt, s)       piece (c, pt) leaves s
//   Policy::evaluate(acc, side_to_move) -> int  centipawns for the side to move
//
// NnueEval (core/nnue.h) follows the same interface; its weights are part
// of EvalTables.
//
// Adding and removing are exact inverses, and every position's accumulator
// equals the one built from scratch (tests check both at every node).

#include "core/nnue.h"
#include "core/pesto_tables.h"
#include "core/types.h"

namespace cb {

// Lookup tables of every evaluation policy. Built once on the host by
// init_eval_tables(); kernels use a copy in device memory.
struct EvalTables {
    // [color][piece type]: piece value, negated for black, so the
    // accumulator is white's material minus black's.
    int material[2][8];
    // [color][piece type][square]: PeSTO middlegame and endgame scores
    // (value + table entry) packed into one int by pack_mg_eg(), negated for
    // black.
    int pesto[2][8][64];
    // [piece type]: game phase contribution (see pesto::kPhaseInc).
    int phase[8];
    // NnueEval's network (core/nnue.h): the engine's default weights.
    NnueWeights nnue;
};

// Middlegame and endgame scores packed into one int, so one addition updates
// both: the endgame score sits in the high 16 bits and the middlegame score
// in the low 16 (borrowing from the high half when negative). Sums stay
// exact while both halves fit in 16 bits; real positions stay far below.
CB_HD constexpr int pack_mg_eg(int mg, int eg) { return int(u32(eg) << 16) + mg; }
CB_HD constexpr int unpack_mg(int packed) { return int(static_cast<short>(u32(packed) & 0xFFFFu)); }
CB_HD constexpr int unpack_eg(int packed) { return int(static_cast<short>((u32(packed) + 0x8000u) >> 16)); }

// Classic piece values (Michniewski's Simplified Evaluation Function).
constexpr int kMaterialValue[8] = {0, 100, 320, 330, 500, 900, 0, 0};

// Fills t (host only).
inline void init_eval_tables(EvalTables& t) {
    for (int c = 0; c < 2; ++c) {
        const int sign = c == WHITE ? 1 : -1;
        for (int pt = 0; pt < 8; ++pt) {
            t.material[c][pt] = sign * kMaterialValue[pt];
            for (int s = 0; s < 64; ++s) {
                int packed = 0;
                if (pt >= PAWN && pt <= KING) {
                    // White reads the table vertically flipped (see pesto_tables.h).
                    const int entry = c == WHITE ? (s ^ 56) : s;
                    const int mg = pesto::kMgValue[pt - 1] + pesto::kMgTables[pt - 1][entry];
                    const int eg = pesto::kEgValue[pt - 1] + pesto::kEgTables[pt - 1][entry];
                    packed = pack_mg_eg(sign * mg, sign * eg);
                }
                t.pesto[c][pt][s] = packed;
            }
        }
    }
    for (int pt = 0; pt < 8; ++pt) t.phase[pt] = (pt >= PAWN && pt <= KING) ? pesto::kPhaseInc[pt - 1] : 0;
    t.nnue = kDefaultNnue;
}

// Material only.
struct MaterialEval {
    static constexpr const char* kName = "material";

    struct Accumulator {
        int material = 0;  // white minus black

        CB_HD bool operator==(const Accumulator& o) const { return material == o.material; }
    };

    CB_HD static void add(Accumulator& acc, const EvalTables& t, Color c, PieceType pt, Square) {
        acc.material += load_readonly(&t.material[c][pt]);
    }
    CB_HD static void remove(Accumulator& acc, const EvalTables& t, Color c, PieceType pt, Square) {
        acc.material -= load_readonly(&t.material[c][pt]);
    }
    CB_HD static int evaluate(const Accumulator& acc, Color side_to_move) {
        return side_to_move == WHITE ? acc.material : -acc.material;
    }
};

// Material + PeSTO piece-square tables, tapered between middlegame and
// endgame by the game phase.
struct PestoEval {
    static constexpr const char* kName = "pesto";

    struct Accumulator {
        int packed = 0;  // pack_mg_eg(mg, eg), white minus black
        int phase = 0;   // sum of pesto::kPhaseInc over all pieces; can exceed 24 after promotions

        CB_HD bool operator==(const Accumulator& o) const { return packed == o.packed && phase == o.phase; }
    };

    CB_HD static void add(Accumulator& acc, const EvalTables& t, Color c, PieceType pt, Square s) {
        acc.packed += load_readonly(&t.pesto[c][pt][s]);
        acc.phase += load_readonly(&t.phase[pt]);
    }
    CB_HD static void remove(Accumulator& acc, const EvalTables& t, Color c, PieceType pt, Square s) {
        acc.packed -= load_readonly(&t.pesto[c][pt][s]);
        acc.phase -= load_readonly(&t.phase[pt]);
    }
    CB_HD static int evaluate(const Accumulator& acc, Color side_to_move) {
        const int mg_phase = acc.phase < pesto::kPhaseMax ? acc.phase : pesto::kPhaseMax;
        const int eg_phase = pesto::kPhaseMax - mg_phase;
        const int white = (unpack_mg(acc.packed) * mg_phase + unpack_eg(acc.packed) * eg_phase) / pesto::kPhaseMax;
        // Division truncates toward zero, so negating afterwards gives exactly
        // the score of the color-flipped position.
        return side_to_move == WHITE ? white : -white;
    }
};

// The evaluation from the side to move's point of view: Eval's
// evaluate(board, accumulator, side) if it has one (terms that read the
// board), else evaluate(accumulator, side).
template <class Eval, class Board>
CB_HD int evaluate_position(const Board& b, const typename Eval::Accumulator& acc, Color side_to_move) {
    if constexpr (requires { Eval::evaluate(b, acc, side_to_move); }) {
        return Eval::evaluate(b, acc, side_to_move);
    } else {
        return Eval::evaluate(acc, side_to_move);
    }
}

// The same with the search's attack lookups, which an eval counting terms
// from the board may use (mobility); other evals ignore them.
template <class Eval, class Board, class Attacks>
CB_HD int evaluate_position(const Board& b, const typename Eval::Accumulator& acc, Color side_to_move,
                            const Attacks& a) {
    if constexpr (requires { Eval::evaluate(b, acc, side_to_move, a); }) {
        return Eval::evaluate(b, acc, side_to_move, a);
    } else {
        return evaluate_position<Eval>(b, acc, side_to_move);
    }
}

// Accumulator of b built from scratch.
template <class Eval, class Board>
CB_HD typename Eval::Accumulator compute_accumulator(const Board& b, const EvalTables& t) {
    typename Eval::Accumulator acc;
    const u64 black = b.color(BLACK);
    for (u64 occ = b.occupied(); occ;) {
        const Square s = Square(pop_lsb(occ));
        Eval::add(acc, t, (black & bit(s)) ? BLACK : WHITE, b.piece_type_on(s), s);
    }
    return acc;
}

}  // namespace cb
