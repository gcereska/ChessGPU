#pragma once

// Shared test helpers: test positions and tree walking.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "host/fen.h"
#include "reference_chess.h"

namespace testutil {

// Positions exercised by most tests: the standard perft suite plus positions
// aimed at specific rules. All are canonical FENs (they round-trip exactly).
inline const std::vector<std::string>& test_fens() {
    static const std::vector<std::string> fens = {
        cb::kStartFen,
        // Standard perft suite (tests/perft/standard.epd).
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        // En passant would expose the king along the rank: exd6 is illegal.
        "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
        // En passant captures the pawn giving check.
        "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
        // En passant available right after the double push.
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        // Double check (knight f3 and rook e8): only king moves.
        "k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1",
        // Rook pinned along the e-file can only move along it.
        "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
        // Castling both ways for both sides.
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        // Castling through and into attacked squares.
        "r3k2r/8/8/8/8/8/6p1/R3K2R w KQkq - 0 1",
        "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
        // Capture-promotion onto a rook square, removing castling rights.
        "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1",
        // Promotions with and without capture, for both sides.
        "1r2k3/P7/8/8/8/8/p7/1R2K3 w - - 0 1",
        // Stalemate and checkmate: no legal moves.
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
        // Halfmove clock near saturation.
        "4k3/8/8/8/8/8/8/4K2R w K - 126 80",
    };
    return fens;
}

inline cb::Position parse_or_die(const std::string& fen) {
    cb::Position p;
    std::string error;
    if (!cb::parse_fen(fen, p, error)) {
        std::printf("test FEN does not parse: %s (%s)\n", fen.c_str(), error.c_str());
        std::abort();
    }
    return p;
}

inline std::vector<cb::u16> sorted_raw(const cb::Move* moves, int count) {
    std::vector<cb::u16> raw;
    for (int i = 0; i < count; ++i) raw.push_back(moves[i].raw());
    std::sort(raw.begin(), raw.end());
    return raw;
}

inline std::vector<cb::u16> sorted_raw(const std::vector<cb::Move>& moves) {
    return sorted_raw(moves.data(), int(moves.size()));
}

// The same position with colors swapped and the board flipped vertically:
// white's pieces become black's on the mirrored squares, and the other side
// is to move. Every evaluation must give it exactly the same score.
inline cb::Position mirror(const cb::Position& p) {
    cb::Position m;
    for (int s = 0; s < 64; ++s) {
        const cb::u8 code = p.squares[s];
        m.squares[s ^ 56] = code ? cb::piece_code(~cb::code_color(code), cb::code_type(code)) : cb::u8(0);
    }
    const unsigned rights = cb::state::castling(p.state);
    const unsigned mirrored_rights = ((rights & 3u) << 2) | ((rights >> 2) & 3u);
    m.state = cb::state::pack(mirrored_rights, cb::state::has_ep(p.state), cb::state::ep_file(p.state),
                              ~cb::state::side(p.state), cb::state::halfmove(p.state));
    m.fullmove = p.fullmove;
    return m;
}

// Depth-first walk over the legal move tree (moves from the reference
// implementation), visiting at most `budget` nodes in a fixed order.
// visit(position, reference_legal_moves) is called for every node.
template <class Visit>
void walk_tree(const cb::Position& p, int depth, long long& budget, Visit& visit) {
    if (budget <= 0) return;
    --budget;
    const ref::Board r = ref::from_position(p);
    const std::vector<cb::Move> moves = ref::legal_moves(r);
    visit(p, moves);
    if (depth == 0) return;
    for (const cb::Move m : moves) {
        if (budget <= 0) return;
        walk_tree(ref::to_position(ref::play(r, m)), depth - 1, budget, visit);
    }
}

// Nodes per position for the reference cross-checks: thorough in optimized
// builds, lighter in Debug where everything is ~10x slower.
#ifdef NDEBUG
constexpr long long kTreeBudget = 40000;
#else
constexpr long long kTreeBudget = 4000;
#endif

}  // namespace testutil
