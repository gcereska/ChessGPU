#pragma once

// Transposition table: results of positions already searched, keyed by
// Zobrist key and shared by every worker of a search (the CPU search, or all
// GPU workers plus the GPU-resident YBWC coordinator). A table is an array of
// 32-byte buckets (one memory sector each), indexed by the key's low bits.
// Two entry layouts, both lockless (docs/ARCHITECTURE.md, Search):
//   TTPacked64  4 entries per bucket, each one 64-bit word: a 16-bit key
//               check (the key's top bits) and the data. An entry is written
//               with one aligned 64-bit store, so it can't tear.
//   TTWide128   2 entries per bucket, each two words: key ^ data and data
//               (Hyatt's trick: a torn write fails the full 64-bit check).
// Both encode an entry's data the same way (TTData, 48 bits), so the search
// handles one 64-bit value per probe, which the warp model broadcasts.
//
// Words are read and written with relaxed atomics: other workers write the
// table concurrently, and a probe may see an old or a new entry, never a mix.
// (Device code uses PTX relaxed loads and stores directly rather than
// <cuda/atomic>: Visual Studio's CUDA dependency scanner stops at that
// header, and every header after it would go untracked.)

#include <atomic>

#include "core/score.h"

namespace cb {

enum class TTBound : u8 { None = 0, Upper = 1, Lower = 2, Exact = 3 };

constexpr u32 kTTGenerations = 64;  // generation counter range (6 bits)

// An entry's data in 48 bits: move 16 | score 16 | depth 8 | bound 2 |
// generation 6. Every real entry has a bound, so raw == 0 means "none".
struct TTData {
    u64 raw = 0;

    CB_HD static TTData make(Move m, int score, int depth, TTBound bound, u32 generation) {
        TTData d;
        d.raw = (u64(m.raw()) << 32) | (u64(u16(short(score))) << 16) | (u64(u8(depth)) << 8) |
                (u64(bound) << 6) | u64(generation & (kTTGenerations - 1));
        return d;
    }
    CB_HD bool hit() const { return raw != 0; }
    CB_HD Move move() const { return Move::from_raw(u16(raw >> 32)); }
    CB_HD int score() const { return int(short(u16(raw >> 16))); }  // node-relative mate distances
    CB_HD int depth() const { return int(u8(raw >> 8)); }
    CB_HD TTBound bound() const { return TTBound((raw >> 6) & 3); }
    CB_HD u32 generation() const { return u32(raw & (kTTGenerations - 1)); }
    CB_HD TTData with_move(Move m) const {
        TTData d;
        d.raw = (raw & ~(u64(0xFFFF) << 32)) | (u64(m.raw()) << 32);
        return d;
    }
};

constexpr u64 kTTDataMask = (u64(1) << 48) - 1;

// Mate scores count plies from the root; the table stores them counted from
// the node, so an entry stays right when the position is reached at another ply.
CB_HD constexpr int tt_score_to(int score, int ply) {
    return score > kScoreMateBound ? score + ply : (score < -kScoreMateBound ? score - ply : score);
}
CB_HD constexpr int tt_score_from(int score, int ply) {
    return score > kScoreMateBound ? score - ply : (score < -kScoreMateBound ? score + ply : score);
}

// The bound an alpha-beta result proves: fail-high, fail-low or exact.
CB_HD constexpr TTBound tt_bound_of(int value, int alpha, int beta) {
    return value >= beta ? TTBound::Lower : (value <= alpha ? TTBound::Upper : TTBound::Exact);
}

// May a stored result end a search of the node with window (alpha, beta)?
CB_HD constexpr bool tt_cuts(TTBound bound, int score, int alpha, int beta) {
    return bound == TTBound::Exact || (bound == TTBound::Lower && score >= beta) ||
           (bound == TTBound::Upper && score <= alpha);
}

// What a probe means for a node (search_task's nodes and the GPU-resident
// YBWC's explicit nodes use the same rules): its move to try first (main
// search only) and, if the stored result decides the node for its window,
// the score to return. A stored result never decides the search root (it
// must return a move) or a node on the previous iteration's PV, and needs at
// least the node's depth (quiescence entries count as depth 0).
struct TTUse {
    Move move;
    bool cut = false;
    int score = 0;
};

template <bool kCutoffs>
CB_HD TTUse tt_use(TTData e, int ply, int depth, bool quiescence, bool on_pv, int alpha, int beta) {
    TTUse u;
    if (!e.hit()) return u;
    if (!quiescence) u.move = e.move();
    if constexpr (kCutoffs) {
        const int s = tt_score_from(e.score(), ply);
        if (ply > 0 && !on_pv && e.depth() >= (quiescence ? 0 : depth) && tt_cuts(e.bound(), s, alpha, beta)) {
            u.cut = true;
            u.score = s;
        }
    }
    return u;
}

// The entry for a node that finished with `value`, searched with window
// (alpha, beta) at `depth` (0 for quiescence) and `ply` from the root.
CB_HD TTData tt_entry(Move best, int value, int ply, int depth, int alpha, int beta, u32 generation) {
    return TTData::make(best, tt_score_to(value, ply), depth > 0 ? depth : 0, tt_bound_of(value, alpha, beta),
                        generation);
}

// One table bucket: 32 bytes, aligned to a memory sector.
struct alignas(32) TTBucket {
    u64 w[4];
};
static_assert(sizeof(TTBucket) == 32, "a bucket is one 32-byte sector");

namespace detail {

CB_HD u64 tt_load(const u64* p) {
#if defined(__CUDA_ARCH__)
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#else
    return std::atomic_ref<u64>(*const_cast<u64*>(p)).load(std::memory_order_relaxed);
#endif
}

CB_HD void tt_store(u64* p, u64 v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(p), "l"((unsigned long long)v) : "memory");
#else
    std::atomic_ref<u64>(*p).store(v, std::memory_order_relaxed);
#endif
}

// How much an entry is worth keeping when a new one needs its slot: deep
// entries of the current search first; each generation of age costs 8 plies.
CB_HD int tt_worth(TTData d, u32 generation) {
    const int age = int((generation - d.generation()) & (kTTGenerations - 1));
    return d.depth() - 8 * age;
}

// Same position: replace it unless the stored entry is from this search,
// much deeper and the new result isn't exact. Keep the old move if the new
// result has none.
CB_HD bool tt_replaces(TTData old, TTData fresh, u32 generation) {
    return fresh.bound() == TTBound::Exact || fresh.depth() + 3 >= old.depth() ||
           old.generation() != (generation & (kTTGenerations - 1));
}

}  // namespace detail

struct TTPacked64 {
    static constexpr const char* kName = "packed64";
    static constexpr int kEntries = 4;

    CB_HD static TTData probe(const TTBucket& b, u64 key) {
        const u64 check = key >> 48;
        CB_UNROLL
        for (int i = 0; i < kEntries; ++i) {
            const u64 w = detail::tt_load(&b.w[i]);
            if (w != 0 && (w >> 48) == check) return TTData{w & kTTDataMask};
        }
        return TTData{};
    }

    CB_HD static void store(TTBucket& b, u64 key, TTData d, u32 generation) {
        const u64 check = key >> 48;
        int slot = -1;
        int worst = 1 << 30;
        CB_UNROLL
        for (int i = 0; i < kEntries; ++i) {
            const u64 w = detail::tt_load(&b.w[i]);
            const TTData old{w & kTTDataMask};
            if (w != 0 && (w >> 48) == check) {  // the same position
                if (!detail::tt_replaces(old, d, generation)) return;
                if (d.move().is_none()) d = d.with_move(old.move());
                detail::tt_store(&b.w[i], (check << 48) | d.raw);
                return;
            }
            const int worth = w == 0 ? -(1 << 30) : detail::tt_worth(old, generation);
            if (worth < worst) {
                worst = worth;
                slot = i;
            }
        }
        detail::tt_store(&b.w[slot], (check << 48) | d.raw);
    }
};

struct TTWide128 {
    static constexpr const char* kName = "wide128";
    static constexpr int kEntries = 2;

    CB_HD static TTData probe(const TTBucket& b, u64 key) {
        CB_UNROLL
        for (int i = 0; i < kEntries; ++i) {
            const u64 k = detail::tt_load(&b.w[2 * i]);
            const u64 d = detail::tt_load(&b.w[2 * i + 1]);
            if (d != 0 && (k ^ d) == key) return TTData{d};
        }
        return TTData{};
    }

    CB_HD static void store(TTBucket& b, u64 key, TTData d, u32 generation) {
        int slot = -1;
        int worst = 1 << 30;
        CB_UNROLL
        for (int i = 0; i < kEntries; ++i) {
            const u64 k = detail::tt_load(&b.w[2 * i]);
            const u64 w = detail::tt_load(&b.w[2 * i + 1]);
            const TTData old{w};
            if (w != 0 && (k ^ w) == key) {  // the same position
                if (!detail::tt_replaces(old, d, generation)) return;
                if (d.move().is_none()) d = d.with_move(old.move());
                slot = i;
                break;
            }
            const int worth = w == 0 ? -(1 << 30) : detail::tt_worth(old, generation);
            if (worth < worst) {
                worst = worth;
                slot = i;
            }
        }
        // Key word first: a reader between the two stores sees a mismatch.
        detail::tt_store(&b.w[2 * slot], key ^ d.raw);
        detail::tt_store(&b.w[2 * slot + 1], d.raw);
    }
};

// No table: feature sets without a TT use this layout, which never stores.
struct TTNone {
    static constexpr const char* kName = "none";
    static constexpr int kEntries = 0;
    CB_HD static TTData probe(const TTBucket&, u64) { return TTData{}; }
    CB_HD static void store(TTBucket&, u64, TTData, u32) {}
};

// Where a table is, for the search that uses it (copied into kernels by value).
struct TTDesc {
    TTBucket* buckets = nullptr;  // host or device memory
    u64 bucket_mask = 0;          // bucket count - 1 (a power of two)
    u32 generation = 0;           // of the current search, below kTTGenerations
};

template <class Layout>
struct TTView {
    TTBucket* buckets;
    u64 mask;
    u32 generation;

    CB_HD explicit TTView(const TTDesc& d) : buckets(d.buckets), mask(d.bucket_mask), generation(d.generation) {}
    CB_HD TTData probe(u64 key) const {
        if constexpr (Layout::kEntries == 0) {
            return TTData{};
        } else {
            return Layout::probe(buckets[key & mask], key);
        }
    }
    CB_HD void store(u64 key, TTData d) const {
        if constexpr (Layout::kEntries != 0) Layout::store(buckets[key & mask], key, d, generation);
    }
};

}  // namespace cb
