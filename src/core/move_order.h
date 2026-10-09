#pragma once

// Move ordering for search. One fixed order, produced two ways:
//
//   StagedPicker  generates moves lazily, stage by stage, from a few
//                 bitboards of state per node (no move list). For the thread
//                 model, where a full list per node would be too big to
//                 keep on chip.
//   ListPicker    generates every move into a list, gives each an order key
//                 and picks the highest key next. The warp model orders this
//                 way; on the CPU it reproduces the warp model's order.
//
// Both produce exactly the order defined by order_key() below, so a search
// visits the same tree whichever picker (or execution model) runs it. Tests
// check the two pickers move by move.
//
// The order, best first:
//   1. the PV move (best move of the previous iteration on this line)
//   2. captures: most valuable victim first (queen ... pawn), then least
//      valuable attacker first (pawn ... king); a pawn capturing onto the
//      last rank promotes to a queen here
//   3. en passant captures
//   4. pawn pushes promoting to a queen
//   5. killer moves, if the search keeps them: up to two quiet moves that
//      recently cut off a sibling at the same ply (the newer first), when
//      they are legal here
//   6. castling (king side first)
//   7. other quiet moves: knights, bishops, rooks, queens, single pawn
//      pushes, double pawn pushes, king
//   8. promotions to knight, bishop or rook (with or without capture)
// Ties within a group go by square number (lower first). Quiescence search
// uses groups 2-4 only, unless in check, when every legal move is tried.

#include "core/movegen.h"

namespace cb {

// What legal move generation needs besides the board, minus the attack map
// of the opponent (king moves test their target square directly instead).
struct CheckInfo {
    u64 pinned;     // our pieces pinned to our king
    u64 checkmask;  // where non-king moves may land: all squares when not in check, the checker and
                    // the squares between in single check, none in double check
    Square king;    // our king
    Color us;

    CB_HD bool in_check() const { return checkmask != ~0ull; }
};

template <class Board, class Attacks>
CB_HD CheckInfo make_check_info(const Board& b, const Attacks& a) {
    CheckInfo ci;
    ci.us = state::side(b.state());
    const Color them = ~ci.us;
    const u64 occupied = b.occupied();
    const u64 our = b.color(ci.us);
    const u64 their = b.color(them);
    ci.king = Square(lsb(b.template pieces<KING>() & our));
    const u64 their_diagonal = b.diagonal_sliders() & their;
    const u64 their_straight = b.straight_sliders() & their;
    const u64 checkers = (a.pawn(ci.us, ci.king) & b.template pieces<PAWN>() & their) |
                         (a.knight(ci.king) & b.template pieces<KNIGHT>() & their) |
                         (a.bishop(ci.king, occupied) & their_diagonal) | (a.rook(ci.king, occupied) & their_straight);
    if (checkers == 0) ci.checkmask = ~0ull;
    else if ((checkers & (checkers - 1)) == 0) ci.checkmask = a.between(ci.king, Square(lsb(checkers))) | checkers;
    else ci.checkmask = 0;

    u64 pinned = 0;
    u64 snipers = (a.rook_empty_board(ci.king) & their_straight) | (a.bishop_empty_board(ci.king) & their_diagonal);
    while (snipers) {
        const Square s = Square(pop_lsb(snipers));
        const u64 blockers = a.between(ci.king, s) & occupied;
        if (blockers && (blockers & (blockers - 1)) == 0) pinned |= blockers & our;
    }
    ci.pinned = pinned;
    return ci;
}

// Pieces of type pt (a runtime value), both colors.
template <class Board>
CB_HD u64 pieces_of_type(const Board& b, PieceType pt) {
    switch (pt) {
        case PAWN: return b.template pieces<PAWN>();
        case KNIGHT: return b.template pieces<KNIGHT>();
        case BISHOP: return b.template pieces<BISHOP>();
        case ROOK: return b.template pieces<ROOK>();
        case QUEEN: return b.template pieces<QUEEN>();
        case KING: return b.template pieces<KING>();
        default: return 0;
    }
}

// True if our king can stand on `to` (after moving from ci.king) without
// being attacked.
template <class Board, class Attacks>
CB_HD bool king_target_safe(const Board& b, const Attacks& a, const CheckInfo& ci, Square to) {
    const u64 occupied = b.occupied() ^ bit(ci.king);
    return (attackers_to(b, a, to, occupied) & b.color(~ci.us)) == 0;
}

namespace order {

// Groups, most important first (higher = earlier). Bits 29-31 of a key.
// No legal move's key is 0 (the warp picker marks picked moves with 0):
// even group 0 keys hold the promoting pawn's square, never h8 or a1.
constexpr u32 kPv = 7;
constexpr u32 kCapture = 6;
constexpr u32 kEnPassant = 5;
constexpr u32 kQueenPromotion = 4;
constexpr u32 kKiller = 3;
constexpr u32 kCastle = 2;
constexpr u32 kQuiet = 1;
constexpr u32 kUnderPromotion = 0;

// Order of quiet moves by kind (higher first).
constexpr u32 kQuietKnight = 6;
constexpr u32 kQuietBishop = 5;
constexpr u32 kQuietRook = 4;
constexpr u32 kQuietQueen = 3;
constexpr u32 kQuietPawnSingle = 2;
constexpr u32 kQuietPawnDouble = 1;
constexpr u32 kQuietKing = 0;

CB_HD constexpr u32 group(u32 key) { return key >> 29; }

CB_HD constexpr u32 quiet_kind(PieceType pt) {
    return pt == KNIGHT   ? kQuietKnight
           : pt == BISHOP ? kQuietBishop
           : pt == ROOK   ? kQuietRook
           : pt == QUEEN  ? kQuietQueen
                          : kQuietKing;
}

// Promotion piece order within group 7: knight, bishop, rook.
CB_HD constexpr u32 under_promotion_rank(PieceType pt) { return pt == KNIGHT ? 2u : pt == BISHOP ? 1u : 0u; }

}  // namespace order

// Moves the search keeps as killers: quiet moves that aren't castling
// (flags QUIET or DOUBLE_PUSH).
CB_HD constexpr bool can_be_killer(Move m) { return m.flags() <= DOUBLE_PUSH; }

// Killer moves of a node (see order_key). Moves compare with their flags, so
// a stored killer only ever equals a quiet move; the pickers still ignore
// killers of any other kind (those keep their own group).
struct Killers {
    Move first;   // the newer
    Move second;

    CB_HD bool has(Move m) const { return !m.is_none() && (m == first || m == second); }
};

// The killers after a cutoff by quiet move m: m becomes the newer killer, the
// newer one the older (unchanged if m already is the newer).
CB_HD constexpr Killers add_killer(Killers k, Move m) { return k.first == m ? k : Killers{m, k.first}; }

// Both killers in one word, for tables shared by concurrent workers (the
// GPU-resident YBWC's explicit nodes).
CB_HD constexpr u32 pack_killers(Killers k) { return u32(k.first.raw()) | (u32(k.second.raw()) << 16); }
CB_HD constexpr Killers unpack_killers(u32 w) {
    return Killers{Move::from_raw(u16(w & 0xFFFFu)), Move::from_raw(u16(w >> 16))};
}

// The ordering key of a legal move: higher keys are searched first, and no
// two legal moves of a position share a key.
template <class Board>
CB_HD u32 order_key(const Board& b, Move m, Move pv, Killers killers = {}) {
    if (m == pv) return 0xFFFFFFFFu;
    const u32 from = m.from();
    const u32 to = m.to();
    const u32 flags = m.flags();
    if (flags == EP_CAPTURE) return (order::kEnPassant << 29) | ((63 - from) << 20);
    if (m.is_promotion() && m.promotion_type() != QUEEN)
        return (order::kUnderPromotion << 29) | ((63 - from) << 20) | ((63 - to) << 14) |
               (order::under_promotion_rank(m.promotion_type()) << 12);
    if (m.is_capture()) {
        const u32 victim = b.piece_type_on(m.to());
        const u32 attacker = b.piece_type_on(m.from());
        return (order::kCapture << 29) | (victim << 26) | ((63 - to) << 20) | ((7 - attacker) << 17) |
               ((63 - from) << 11);
    }
    if (flags == PROMO_QUEEN) return (order::kQueenPromotion << 29) | ((63 - from) << 20);
    if (flags == KING_CASTLE || flags == QUEEN_CASTLE)
        return (order::kCastle << 29) | (flags == KING_CASTLE ? (1u << 28) : 0u);
    if (killers.has(m)) return (order::kKiller << 29) | (m == killers.first ? (1u << 28) : 0u);
    const PieceType pt = b.piece_type_on(m.from());
    const u32 kind = pt == PAWN ? (flags == DOUBLE_PUSH ? order::kQuietPawnDouble : order::kQuietPawnSingle)
                                : order::quiet_kind(pt);
    return (order::kQuiet << 29) | (kind << 26) | ((63 - from) << 20) | ((63 - to) << 14);
}

// True for moves quiescence search tries when not in check (groups 2-4).
CB_HD constexpr bool is_quiescence_group(u32 key) {
    const u32 g = order::group(key);
    return g == order::kCapture || g == order::kEnPassant || g == order::kQueenPromotion;
}

// Which moves a node tries.
enum class PickMode : u8 {
    All,          // every legal move (main search, and quiescence when in check)
    Quiescence,   // groups 2-4 only (quiescence when not in check)
    // Groups 2-4, then the quiet moves (not castling) that give check, in
    // their usual order: quiescence's first ply when not in check, with
    // feature sets that have kQChecks.
    QuiescenceChecks,
};

// A quiet move (not castling) that gives check: what QuiescenceChecks adds.
template <class Board, class Attacks>
CB_HD bool is_quiet_check(const Board& b, const Attacks& a, Move m) {
    return m.flags() <= DOUBLE_PUSH && quiet_move_gives_check(b, m, a);
}

// ---------------------------------------------------------------------------
// StagedPicker
// ---------------------------------------------------------------------------

struct StagedPicker {
    enum Stage : u8 {
        kStagePv,
        kStageCaptures,
        kStageEnPassant,
        kStageQueenPromotions,
        kStageKillers,
        kStageCastling,
        kStageQuiets,
        kStageUnderPromotions,
        kStageDone,
    };

    // Per-node state. What set_a, set_b and aux hold depends on the stage:
    //   captures:        aux = victim type | victim square << 3; set_a = victims of that type not
    //                    yet visited; set_b = legal attackers of the current victim not yet tried
    //   en passant:      set_b = pawns that may capture
    //   queen promotion: set_b = pawns that may push to the last rank
    //   killers:         aux = next killer to try (0 first, 1 second, 2 done)
    //   castling:        aux = next side to try (0 king side, 1 queen side, 2 done)
    //   quiets:          aux = kind | from << 3; set_a = pieces of that kind left; set_b = targets
    //                    left (for pawn pushes: target squares, the pawn stands behind)
    //   under-promotion: aux = from | promotion index << 6; set_a = promoting pawns left; set_b =
    //                    targets of the current pawn left
    struct State {
        u64 set_a;
        u64 set_b;
        u16 aux;
        u8 stage;
        u8 mode;          // PickMode
        u16 killers[2];   // raw moves (Killers), in the padding the struct had
    };

    template <class Board, class Attacks>
    CB_HD static void init(State& st, const Board&, const Attacks&, const CheckInfo&, PickMode mode,
                           Killers killers = {}) {
        st.set_a = 0;
        st.set_b = 0;
        st.aux = 0;
        st.mode = u8(mode);
        // Killers only order the main search (quiescence's quiet checks keep
        // their usual order, as in the list picker).
        st.killers[0] = mode == PickMode::All ? killers.first.raw() : u16(0);
        st.killers[1] = mode == PickMode::All ? killers.second.raw() : u16(0);
        // The PV stage also sets up the captures; it only returns the PV move
        // in PickMode::All.
        st.stage = kStagePv;
    }

    // The next move in order, or the empty move when there are none left.
    // `pv` is the node's PV move (empty if none); it is tried first in
    // PickMode::All and skipped where it would otherwise appear.
    template <class Board, class Attacks>
    CB_HD static Move next(State& st, const Board& b, const Attacks& a, const CheckInfo& ci, Move pv) {
        const Color us = ci.us;
        const Color them = ~us;
        const u64 occupied = b.occupied();
        const u64 our = b.color(us);
        const u64 their = b.color(them);
        const u64 empty = ~occupied;
        const int up = forward_offset(us);
        const u64 last_rank = us == WHITE ? kRank8 : kRank1;
        const u64 seventh_rank = us == WHITE ? rank_mask(6) : rank_mask(1);
        const u64 our_pawns = b.template pieces<PAWN>() & our;
        // Pawns that may push: unpinned, or pinned along the king's file.
        const u64 pushable = our_pawns & (~ci.pinned | file_mask(file_of(ci.king)));

        for (;;) {
            switch (st.stage) {
                case kStagePv: {
                    st.stage = kStageCaptures;
                    st.aux = QUEEN;
                    st.set_a = their & b.template pieces<QUEEN>();
                    st.set_b = 0;
                    if (st.mode == u8(PickMode::All) && !pv.is_none() && pv_is_legal(b, a, pv)) return pv;
                    break;
                }

                case kStageCaptures: {
                    if (st.set_b) {
                        const Square victim = Square(st.aux >> 3);
                        const Square from = lowest_value_attacker(b, st.set_b);
                        st.set_b ^= bit(from);
                        const bool promotes = (bit(from) & our_pawns) && (bit(victim) & last_rank);
                        const Move m(from, victim, promotes ? PROMO_CAPTURE_QUEEN : CAPTURE);
                        if (m != pv) return m;
                        break;
                    }
                    if (st.set_a) {
                        const Square victim = Square(pop_lsb(st.set_a));
                        st.set_b = legal_attackers(b, a, ci, victim);
                        st.aux = u16((st.aux & 7) | (u32(victim) << 3));
                        break;
                    }
                    const int victim_type = st.aux & 7;
                    if (victim_type > PAWN) {
                        st.aux = u16(victim_type - 1);
                        st.set_a = their & pieces_of_type(b, PieceType(victim_type - 1));
                        break;
                    }
                    st.stage = kStageEnPassant;
                    st.set_b = state::has_ep(b.state()) ? a.pawn(them, state::ep_square(b.state())) & our_pawns : 0;
                    break;
                }

                case kStageEnPassant: {
                    if (st.set_b) {
                        const Square from = Square(pop_lsb(st.set_b));
                        const Square to = state::ep_square(b.state());
                        LegalContext c;  // en_passant_is_legal reads only these fields
                        c.us = us;
                        c.occupied = occupied;
                        c.their = their;
                        c.king = ci.king;
                        const Move m(from, to, EP_CAPTURE);
                        if (m != pv && en_passant_is_legal(b, a, c, from, to)) return m;
                        break;
                    }
                    st.stage = kStageQueenPromotions;
                    // Pawns whose push square is empty and allowed; from = to - up.
                    st.set_b = shift_forward(pushable & seventh_rank, us) & empty & ci.checkmask;
                    break;
                }

                case kStageQueenPromotions: {
                    if (st.set_b) {
                        const Square to = Square(pop_lsb(st.set_b));
                        const Move m(Square(to - up), to, PROMO_QUEEN);
                        if (m != pv) return m;
                        break;
                    }
                    if (st.mode == u8(PickMode::Quiescence) && !ci.in_check()) {
                        st.stage = kStageDone;
                        break;
                    }
                    if (st.mode == u8(PickMode::QuiescenceChecks) && !ci.in_check()) {
                        // Straight to the quiet moves, which only checks pass.
                        st.stage = kStageQuiets;
                        start_quiet_kind(st, b, a, ci, order::kQuietKnight);
                        break;
                    }
                    st.stage = kStageKillers;
                    st.aux = 0;
                    break;
                }

                case kStageKillers: {
                    if (st.aux < 2) {
                        const Move k = Move::from_raw(st.killers[st.aux]);
                        ++st.aux;
                        // Each killer once, after the PV move, if it is a legal quiet move here.
                        if (!k.is_none() && can_be_killer(k) && k != pv &&
                            (st.aux == 1 || k != Move::from_raw(st.killers[0])) && pv_is_legal(b, a, k))
                            return k;
                        break;
                    }
                    st.stage = kStageCastling;
                    st.aux = 0;
                    break;
                }

                case kStageCastling: {
                    if (st.aux < 2 && !ci.in_check()) {
                        const bool king_side = st.aux == 0;
                        ++st.aux;
                        const Move m = castling_move(b, a, ci, king_side);
                        if (!m.is_none() && m != pv) return m;
                        break;
                    }
                    st.stage = kStageQuiets;
                    start_quiet_kind(st, b, a, ci, order::kQuietKnight);
                    break;
                }

                case kStageQuiets: {
                    const u32 kind = st.aux & 7;
                    if (st.set_b) {
                        const Square to = Square(pop_lsb(st.set_b));
                        Move m;
                        if (kind == order::kQuietPawnSingle) {
                            m = Move(Square(to - up), to, QUIET);
                        } else if (kind == order::kQuietPawnDouble) {
                            m = Move(Square(to - 2 * up), to, DOUBLE_PUSH);
                        } else {
                            const Square from = Square(st.aux >> 3);
                            if (kind == order::kQuietKing && !king_target_safe(b, a, ci, to)) break;
                            m = Move(from, to, QUIET);
                        }
                        if (m != pv && m.raw() != st.killers[0] && m.raw() != st.killers[1] &&
                            (st.mode != u8(PickMode::QuiescenceChecks) || quiet_move_gives_check(b, m, a)))
                            return m;
                        break;
                    }
                    if (st.set_a) {
                        // Next piece of the current kind (pawn kinds never get here).
                        const Square from = Square(pop_lsb(st.set_a));
                        st.aux = u16(kind | (u32(from) << 3));
                        st.set_b = quiet_targets(b, a, ci, kind, from);
                        if (st.mode == u8(PickMode::QuiescenceChecks)) st.set_b &= check_targets(b, a, ci, kind, from);
                        break;
                    }
                    if (kind > order::kQuietKing) {
                        start_quiet_kind(st, b, a, ci, kind - 1);
                        break;
                    }
                    if (st.mode == u8(PickMode::QuiescenceChecks)) {
                        st.stage = kStageDone;  // no under-promotions in quiescence
                        break;
                    }
                    st.stage = kStageUnderPromotions;
                    st.set_a = our_pawns & seventh_rank;
                    st.set_b = 0;
                    st.aux = 0;
                    break;
                }

                case kStageUnderPromotions: {
                    if (st.set_b) {
                        const Square from = Square(st.aux & 63);
                        const u32 index = st.aux >> 6;  // 0 knight, 1 bishop, 2 rook
                        const Square to = Square(lsb(st.set_b));
                        if (index == 2) {
                            st.set_b &= st.set_b - 1;
                            st.aux = u16(from);
                        } else {
                            st.aux = u16(u32(from) | ((index + 1) << 6));
                        }
                        const bool capture = (bit(to) & their) != 0;
                        const u32 promo = (capture ? PROMO_CAPTURE_KNIGHT : PROMO_KNIGHT) + index;
                        const Move m(from, to, promo);
                        if (m != pv) return m;
                        break;
                    }
                    if (st.set_a) {
                        const Square from = Square(pop_lsb(st.set_a));
                        st.aux = u16(from);
                        st.set_b = promotion_targets(b, a, ci, from);
                        break;
                    }
                    st.stage = kStageDone;
                    break;
                }

                default:
                    return Move();
            }
        }
    }

  private:
    // The PV move comes from the previous iteration's line, so it is legal
    // whenever the line was followed exactly; checked anyway, so a wrong hint
    // can only cost ordering, never correctness. Killers, made in sibling
    // positions, are checked the same way: a move must equal a generated
    // move exactly, flags included.
    template <class Board, class Attacks>
    CB_HD static bool pv_is_legal(const Board& b, const Attacks& a, Move pv) {
        if (!(b.color(state::side(b.state())) & bit(pv.from()))) return false;
        const LegalContext c = make_legal_context(b, a);
        MoveBuffer<32> moves;
        generate_moves_from(b, a, c, pv.from(), moves);
        for (int i = 0; i < moves.size; ++i)
            if (moves.moves[i] == pv) return true;
        return false;
    }

    // Lowest-valued piece in `set` (pawn first, king last), lowest square on ties.
    template <class Board>
    CB_HD static Square lowest_value_attacker(const Board& b, u64 set) {
        u64 m = set & b.template pieces<PAWN>();
        if (!m) m = set & b.template pieces<KNIGHT>();
        if (!m) m = set & b.template pieces<BISHOP>();
        if (!m) m = set & b.template pieces<ROOK>();
        if (!m) m = set & b.template pieces<QUEEN>();
        if (!m) m = set;  // the king
        return Square(lsb(m));
    }

    // Our pieces that can legally capture on `victim`.
    template <class Board, class Attacks>
    CB_HD static u64 legal_attackers(const Board& b, const Attacks& a, const CheckInfo& ci, Square victim) {
        const u64 occupied = b.occupied();
        const u64 our = b.color(ci.us);
        u64 attackers = 0;
        if (bit(victim) & ci.checkmask) {
            attackers = ((a.pawn(~ci.us, victim) & b.template pieces<PAWN>()) |
                         (a.knight(victim) & b.template pieces<KNIGHT>()) |
                         (a.bishop(victim, occupied) & b.diagonal_sliders()) |
                         (a.rook(victim, occupied) & b.straight_sliders())) &
                        our;
            // A pinned piece may only capture along its pin line.
            for (u64 p = attackers & ci.pinned; p;) {
                const Square from = Square(pop_lsb(p));
                if (!(a.line(ci.king, from) & bit(victim))) attackers ^= bit(from);
            }
        }
        if ((a.king(victim) & bit(ci.king)) && king_target_safe(b, a, ci, victim)) attackers |= bit(ci.king);
        return attackers;
    }

    template <class Board, class Attacks>
    CB_HD static Move castling_move(const Board& b, const Attacks& a, const CheckInfo& ci, bool king_side) {
        const u32 rights = state::castling(b.state());
        const int base = ci.us == WHITE ? 0 : 56;
        const u32 right = king_side ? (ci.us == WHITE ? WHITE_OO : BLACK_OO) : (ci.us == WHITE ? WHITE_OOO : BLACK_OOO);
        if (!(rights & right)) return Move();
        // Castling rights guarantee the king and rook stand on their squares.
        const u64 must_be_empty = (king_side ? (bit(F1) | bit(G1)) : (bit(B1) | bit(C1) | bit(D1))) << base;
        if (b.occupied() & must_be_empty) return Move();
        const Square pass = Square((king_side ? F1 : D1) + base);
        const Square land = Square((king_side ? G1 : C1) + base);
        const u64 their = b.color(~ci.us);
        // The king isn't in check (tested by the caller), so the king still
        // standing on its square can't hide an attack on these two.
        if (attackers_to(b, a, pass, b.occupied()) & their) return Move();
        if (attackers_to(b, a, land, b.occupied()) & their) return Move();
        return Move(ci.king, land, king_side ? KING_CASTLE : QUEEN_CASTLE);
    }

    // Quiet targets of the piece on `from`, of quiet kind `kind` (not pawns).
    // King targets still need king_target_safe().
    template <class Board, class Attacks>
    CB_HD static u64 quiet_targets(const Board& b, const Attacks& a, const CheckInfo& ci, u32 kind, Square from) {
        const u64 occupied = b.occupied();
        const u64 empty = ~occupied;
        if (kind == order::kQuietKing) return a.king(from) & empty;
        u64 attacks;
        switch (kind) {
            case order::kQuietKnight: attacks = a.knight(from); break;
            case order::kQuietBishop: attacks = a.bishop(from, occupied); break;
            case order::kQuietRook: attacks = a.rook(from, occupied); break;
            default: attacks = a.queen(from, occupied); break;
        }
        const u64 pin_line = (bit(from) & ci.pinned) ? a.line(ci.king, from) : ~0ull;
        return attacks & empty & ci.checkmask & pin_line;
    }

    // PickMode::QuiescenceChecks: the squares the piece on `from` (quiet kind
    // `kind`, not pawns) may give check from. A piece that blocks one of our
    // sliders from their king may go anywhere (leaving the line uncovers the
    // slider); any other piece checks only from a square it would attack the
    // king from (the board as it is: if it moved along a line toward that
    // king, what blocks it now still would). quiet_move_gives_check() still
    // decides each move; this only spares testing the others.
    template <class Board, class Attacks>
    CB_HD static u64 check_targets(const Board& b, const Attacks& a, const CheckInfo& ci, u32 kind, Square from) {
        const Square king = king_square(b, ~ci.us);
        const u64 occupied = b.occupied();
        if (a.line(king, from) && slider_sees_king(b, a, ci.us, king, from, occupied ^ bit(from))) return ~0ull;
        switch (kind) {
            case order::kQuietKnight: return a.knight(king);
            case order::kQuietBishop: return a.bishop(king, occupied);
            case order::kQuietRook: return a.rook(king, occupied);
            case order::kQuietQueen: return a.queen(king, occupied);
            default: return 0;  // a king checks only by uncovering
        }
    }

    // Sets up quiet kind `kind`: pieces to visit, or all push targets at once for pawns.
    template <class Board, class Attacks>
    CB_HD static void start_quiet_kind(State& st, const Board& b, const Attacks& a, const CheckInfo& ci, u32 kind) {
        st.aux = u16(kind);
        st.set_a = 0;
        st.set_b = 0;
        const Color us = ci.us;
        const u64 our = b.color(us);
        const bool double_check = ci.checkmask == 0;
        if (kind == order::kQuietPawnSingle || kind == order::kQuietPawnDouble) {
            const u64 empty = ~b.occupied();
            const u64 last_rank = us == WHITE ? kRank8 : kRank1;
            const u64 pawns = b.template pieces<PAWN>() & our;
            const u64 pushable = pawns & (~ci.pinned | file_mask(file_of(ci.king)));
            // QuiescenceChecks: pushes may check from the squares a pawn
            // attacks their king from, or by uncovering a slider (a pawn that
            // blocks one is among the pieces that king sees). Pushes stay on
            // the file, so quiet_move_gives_check() still decides each one.
            u64 checks = ~0ull;
            if (st.mode == u8(PickMode::QuiescenceChecks)) {
                const Square king = king_square(b, ~us);
                u64 blockers = 0;
                for (u64 seen = a.queen(king, b.occupied()) & pushable; seen;) {
                    const Square from = Square(pop_lsb(seen));
                    if (slider_sees_king(b, a, us, king, from, b.occupied() ^ bit(from))) blockers |= bit(from);
                }
                const u64 moved = kind == order::kQuietPawnSingle ? shift_forward(blockers, us)
                                                                  : shift_forward(shift_forward(blockers, us), us);
                checks = a.pawn(~us, king) | moved;
            }
            const u64 single = shift_forward(pushable, us) & empty;
            if (kind == order::kQuietPawnSingle) {
                st.set_b = single & ~last_rank & ci.checkmask & checks;
            } else {
                const u64 double_push_rank = us == WHITE ? rank_mask(3) : rank_mask(4);
                st.set_b = shift_forward(single, us) & empty & double_push_rank & ci.checkmask & checks;
            }
            return;
        }
        if (double_check && kind != order::kQuietKing) return;
        switch (kind) {
            case order::kQuietKnight: st.set_a = b.template pieces<KNIGHT>() & our; break;
            case order::kQuietBishop: st.set_a = b.template pieces<BISHOP>() & our; break;
            case order::kQuietRook: st.set_a = b.template pieces<ROOK>() & our; break;
            case order::kQuietQueen: st.set_a = b.template pieces<QUEEN>() & our; break;
            default: st.set_a = bit(ci.king); break;
        }
    }

    // Squares the pawn on `from` (on the seventh rank) can legally promote on.
    template <class Board, class Attacks>
    CB_HD static u64 promotion_targets(const Board& b, const Attacks& a, const CheckInfo& ci, Square from) {
        const Color us = ci.us;
        const u64 pin_line = (bit(from) & ci.pinned) ? a.line(ci.king, from) : ~0ull;
        const u64 push = shift_forward(bit(from), us) & ~b.occupied();
        const u64 captures = a.pawn(us, from) & b.color(~us);
        return (push | captures) & ci.checkmask & pin_line;
    }
};

// ---------------------------------------------------------------------------
// ListPicker
// ---------------------------------------------------------------------------

// Moves and keys of the nodes on the current search path, one segment per
// node, stacked. kCapacity bounds the total over the path.
template <int kCapacity>
struct MoveArena {
    static constexpr int capacity = kCapacity;
    Move moves[kCapacity];
    u32 keys[kCapacity];
};

struct ListPicker {
    struct State {
        u16 begin;   // first entry of this node's segment
        u16 end;     // one past the last entry
        u16 picked;  // entries before begin + picked are already tried
        u8 mode;
        u8 overflow;  // the arena was too small for this node's moves
    };

    // Generates the node's moves into the arena after `arena_top` (the end
    // of the parent's segment) and returns the new top.
    template <class Board, class Attacks, int kCapacity>
    CB_HD static int init(State& st, MoveArena<kCapacity>& arena, int arena_top, const Board& b, const Attacks& a,
                          PickMode mode, Move pv, Killers killers = {}) {
        MoveList list;
        generate_legal(b, a, list);
        st.begin = u16(arena_top);
        st.picked = 0;
        st.mode = u8(mode);
        st.overflow = 0;
        int top = arena_top;
        for (int i = 0; i < list.size; ++i) {
            const u32 key = order_key(b, list.moves[i], mode == PickMode::All ? pv : Move(),
                                      mode == PickMode::All ? killers : Killers{});
            if (mode == PickMode::Quiescence && !is_quiescence_group(key)) continue;
            if (mode == PickMode::QuiescenceChecks && !is_quiescence_group(key) && !is_quiet_check(b, a, list.moves[i]))
                continue;
            if (top == kCapacity) {
                st.overflow = 1;
                break;
            }
            arena.moves[top] = list.moves[i];
            arena.keys[top] = key;
            ++top;
        }
        st.end = u16(top);
        return top;
    }

    // ABDADA (see core/abdada.h): the key a deferred move gets back. No legal
    // move's key is this low (see order::kPv), so deferred moves come after
    // every other one, in the order they were deferred.
    static constexpr u32 kDeferredKey = 1;

    // Puts the move next() just returned back among the untried ones, with
    // kDeferredKey.
    template <int kCapacity>
    CB_HD static void defer(State& st, MoveArena<kCapacity>& arena) {
        --st.picked;
        arena.keys[st.begin + st.picked] = kDeferredKey;
    }

    // Whether the move next() just returned had been deferred.
    template <int kCapacity>
    CB_HD static bool returned_deferred(const State& st, const MoveArena<kCapacity>& arena) {
        return st.picked > 0 && arena.keys[st.begin + st.picked - 1] == kDeferredKey;
    }

    // Highest-keyed move not tried yet, or the empty move.
    template <int kCapacity>
    CB_HD static Move next(State& st, MoveArena<kCapacity>& arena) {
        const int first = st.begin + st.picked;
        if (first >= st.end) return Move();
        int best = first;
        for (int i = first + 1; i < st.end; ++i)
            if (arena.keys[i] > arena.keys[best]) best = i;
        const Move m = arena.moves[best];
        const u32 k = arena.keys[best];
        arena.moves[best] = arena.moves[first];
        arena.keys[best] = arena.keys[first];
        arena.moves[first] = m;
        arena.keys[first] = k;
        ++st.picked;
        return m;
    }
};

}  // namespace cb
