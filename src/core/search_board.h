#pragma once

// A board plus the state search keeps incrementally next to it: the Zobrist
// key and the evaluation accumulator.

#include "core/eval.h"
#include "core/zobrist.h"

namespace cb {

// Tables search needs besides the attack tables. Built once on the host by
// init_search_tables(); kernels use a copy in device memory.
struct SearchTables {
    ZobristKeys zobrist;
    EvalTables eval;
};

inline void init_search_tables(SearchTables& t) {
    init_zobrist_keys(t.zobrist);
    init_eval_tables(t.eval);
}

template <class Board, class Eval>
struct SearchBoard {
    Board board;
    u64 key = 0;
    typename Eval::Accumulator acc;
};

namespace detail {

template <class Eval>
struct KeyEvalObserver {
    u64& key;
    typename Eval::Accumulator& acc;
    const SearchTables& t;

    CB_HD void add(Color c, PieceType pt, Square s) {
        key ^= piece_key(t.zobrist, c, pt, s);
        Eval::add(acc, t.eval, c, pt, s);
    }
    CB_HD void remove(Color c, PieceType pt, Square s) {
        key ^= piece_key(t.zobrist, c, pt, s);
        Eval::remove(acc, t.eval, c, pt, s);
    }
};

template <class Eval>
struct EvalObserver {
    typename Eval::Accumulator& acc;
    const EvalTables& t;

    CB_HD void add(Color c, PieceType pt, Square s) { Eval::add(acc, t, c, pt, s); }
    CB_HD void remove(Color c, PieceType pt, Square s) { Eval::remove(acc, t, c, pt, s); }
};

}  // namespace detail

// Key and accumulator built from scratch.
template <class Eval, class Board, class Attacks>
CB_HD SearchBoard<Board, Eval> make_search_board(const Board& b, const Attacks& a, const SearchTables& t) {
    SearchBoard<Board, Eval> sb;
    sb.board = b;
    sb.key = compute_key(b, a, t.zobrist);
    sb.acc = compute_accumulator<Eval>(b, t.eval);
    return sb;
}

// Plays a legal move, updating the key and the accumulator.
template <class Board, class Eval, class Attacks>
CB_HD Undo make_search_move(SearchBoard<Board, Eval>& sb, Move m, const Attacks& a, const SearchTables& t) {
    const u32 old_state = sb.board.state();
    // Both en passant tests return at once unless the state has an en
    // passant square, which only happens right after a double push.
    const bool old_ep = en_passant_capture_possible(sb.board, a);
    detail::KeyEvalObserver<Eval> obs{sb.key, sb.acc, t};
    const Undo undo = make_move(sb.board, m, obs);
    const u32 new_state = sb.board.state();
    sb.key ^= state_key(t.zobrist, old_state, old_ep) ^
              state_key(t.zobrist, new_state, en_passant_capture_possible(sb.board, a));
    return undo;
}

// Takes back m, the last move played. The key isn't recomputed: search keeps
// every position's key on its stack (for repetition detection) and passes
// the one from before the move.
template <class Board, class Eval>
CB_HD void unmake_search_move(SearchBoard<Board, Eval>& sb, Move m, Undo undo, u64 key_before,
                              const SearchTables& t) {
    detail::EvalObserver<Eval> obs{sb.acc, t.eval};
    unmake_move(sb.board, m, undo, obs);
    sb.key = key_before;
}

// Passes the move (see make_null_move): the key changes with the state word;
// the accumulator stays (evaluate takes the side to move).
template <class Board, class Eval, class Attacks>
CB_HD Undo make_null_search_move(SearchBoard<Board, Eval>& sb, const Attacks& a, const SearchTables& t) {
    const u32 old_state = sb.board.state();
    const bool old_ep = en_passant_capture_possible(sb.board, a);
    const Undo undo = make_null_move(sb.board);
    sb.key ^= state_key(t.zobrist, old_state, old_ep) ^ state_key(t.zobrist, sb.board.state(), false);
    return undo;
}

template <class Board, class Eval>
CB_HD void unmake_null_search_move(SearchBoard<Board, Eval>& sb, Undo undo, u64 key_before) {
    unmake_null_move(sb.board, undo);
    sb.key = key_before;
}

// True if `side` has a piece other than pawns and the king: then zugzwang,
// where passing would beat every move, is rare enough to try null moves.
template <class Board>
CB_HD bool has_non_pawn_material(const Board& b, Color side) {
    return ((b.template pieces<KNIGHT>() | b.diagonal_sliders() | b.straight_sliders()) & b.color(side)) != 0;
}

// True for positions no sequence of moves can checkmate in: king against
// king, or against king and one minor piece.
template <class Board>
CB_HD bool insufficient_material(const Board& b) {
    if (b.template pieces<PAWN>() | b.straight_sliders()) return false;
    const u64 minors = b.template pieces<KNIGHT>() | b.template pieces<BISHOP>();
    return (minors & (minors - 1)) == 0;
}

}  // namespace cb
