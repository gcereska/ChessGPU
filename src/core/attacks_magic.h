#pragma once

// Magic bitboards for sliding pieces (docs/ARCHITECTURE.md, Move generation):
// the blockers on a piece's lines (excluding the board's edge, where a
// blocker changes nothing) times a per-square "magic" number put a perfect
// index into the top bits, and a table holds the attack set for every
// blocker pattern. "Fancy" magics: each square has its own table size
// (2^bits for its relevant squares), 107,648 entries (841 KiB) in all.
// The magics are searched on the host when the tables are built (a fixed
// seed, so every build finds the same ones).

#include <vector>

#include "core/attacks.h"

namespace cb {

struct MagicEntry {
    u64 mask;    // relevant blockers: the piece's lines without the edge squares
    u64 magic;
    int offset;  // of the square's first entry in MagicTables::attacks
    int shift;   // 64 - number of relevant squares
};

constexpr int kRookMagicEntries = 102400;
constexpr int kBishopMagicEntries = 5248;

struct MagicTables {
    MagicEntry rook[64];
    MagicEntry bishop[64];
    u64 attacks[kRookMagicEntries + kBishopMagicEntries];
};

namespace detail {

// xorshift64*: a fixed sequence, so every build finds the same magics.
struct MagicRng {
    u64 s;
    u64 next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ull;
    }
    u64 sparse() { return next() & next() & next(); }  // few bits set: good magic candidates
};

// Rook (straight) or bishop (diagonal) attacks from s, by walking rays.
inline u64 slider_attacks_by_rays(Square s, u64 occupied, bool straight) {
    static constexpr int kStraight[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static constexpr int kDiagonal[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    u64 result = 0;
    for (const auto& d : straight ? kStraight : kDiagonal) result |= walk_ray(s, d[0], d[1], occupied);
    return result;
}

// The relevant blockers of a slider on s: its empty-board attacks minus the
// last square of each ray (nothing lies beyond it).
inline u64 relevant_mask(Square s, bool straight) {
    static constexpr int kStraight[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static constexpr int kDiagonal[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    u64 mask = 0;
    for (const auto& d : straight ? kStraight : kDiagonal) {
        int f = file_of(s) + d[0];
        int r = rank_of(s) + d[1];
        while (on_board(f + d[0], r + d[1])) {  // stop before the ray's last square
            mask |= bit(make_square(f, r));
            f += d[0];
            r += d[1];
        }
    }
    return mask;
}

// Finds a magic for one square and fills its part of `table` (from `offset`).
inline MagicEntry find_magic(Square s, bool straight, int offset, u64* table, MagicRng& rng) {
    MagicEntry e{};
    e.mask = relevant_mask(s, straight);
    const int bits = popcount(e.mask);
    e.shift = 64 - bits;
    e.offset = offset;
    const int size = 1 << bits;
    // Every blocker subset (carry-rippler) and its attack set.
    std::vector<u64> occupancy(size), reference(size);
    u64 subset = 0;
    for (int i = 0; i < size; ++i) {
        occupancy[i] = subset;
        reference[i] = slider_attacks_by_rays(s, subset, straight);
        subset = (subset - e.mask) & e.mask;
    }
    std::vector<int> used(size, 0);  // try number that filled each slot
    for (int attempt = 1;; ++attempt) {
        const u64 magic = rng.sparse();
        if (popcount((e.mask * magic) & 0xFF00000000000000ull) < 6) continue;  // spreads too few bits up
        bool ok = true;
        for (int i = 0; i < size && ok; ++i) {
            const int index = int((occupancy[i] * magic) >> e.shift);
            if (used[index] != attempt) {
                used[index] = attempt;
                table[offset + index] = reference[i];
            } else if (table[offset + index] != reference[i]) {
                ok = false;  // two patterns with different attacks share a slot
            }
        }
        if (ok) {
            e.magic = magic;
            return e;
        }
    }
}

}  // namespace detail

// Fills m (host only; a few tens of milliseconds).
inline void init_magic_tables(MagicTables& m) {
    detail::MagicRng rng{0x9E3779B97F4A7C15ull};
    int offset = 0;
    for (int i = 0; i < 64; ++i) {
        m.rook[i] = detail::find_magic(Square(i), true, offset, m.attacks, rng);
        offset += 1 << (64 - m.rook[i].shift);
    }
    for (int i = 0; i < 64; ++i) {
        m.bishop[i] = detail::find_magic(Square(i), false, offset, m.attacks, rng);
        offset += 1 << (64 - m.bishop[i].shift);
    }
}

class MagicAttacks : public TableAttacks {
  public:
    static constexpr const char* kName = "magic";

    CB_HD MagicAttacks(const AttackTables& tables, const MagicTables& magic) : TableAttacks(tables), m_(&magic) {}

    CB_HD u64 bishop(Square s, u64 occupied) const { return lookup(m_->bishop[s], occupied); }
    CB_HD u64 rook(Square s, u64 occupied) const { return lookup(m_->rook[s], occupied); }
    CB_HD u64 queen(Square s, u64 occupied) const { return bishop(s, occupied) | rook(s, occupied); }

  private:
    CB_HD u64 lookup(const MagicEntry& e, u64 occupied) const {
        const u64 index = ((occupied & load_readonly(&e.mask)) * load_readonly(&e.magic)) >> load_readonly(&e.shift);
        return load_readonly(&m_->attacks[load_readonly(&e.offset) + int(index)]);
    }

    const MagicTables* m_;
};

}  // namespace cb
