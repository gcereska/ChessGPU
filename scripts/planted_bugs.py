"""Planted-bug campaigns: each plants one bug at a time in the sources,
rebuilds, runs the tests that should catch it, restores the file, and logs
whether the bug was caught (tests failed or hung) or missed. Misses must be
explained in the test files (for example, mutations that can't change any
result).

  python scripts/planted_bugs.py killers-cpu    # killer moves, CPU side (build/mut-build-cpu, target unit_tests)
  python scripts/planted_bugs.py see-cpu        # SEE and quiescence pruning, CPU side
  python scripts/planted_bugs.py abdada-cpu     # ABDADA's shared code (claims, deferral, hooks), CPU side
  python scripts/planted_bugs.py move32-cpu     # 32-bit moves (core/move32.h), CPU side
  python scripts/planted_bugs.py nnue-cpu       # NNUE evaluation and trainer, CPU side
  python scripts/planted_bugs.py pvs-cpu        # principal variation search, CPU side
  python scripts/planted_bugs.py qchecks-cpu    # checks in quiescence, CPU side
  python scripts/planted_bugs.py aspiration-cpu # aspiration windows and the fail-low time extension, CPU side
  python scripts/planted_bugs.py multipv-cpu    # MultiPV (host/multipv.cpp), CPU side
  python scripts/planted_bugs.py gpu-explicit   # killers and SEE in GPU-only code (build/mut-build; needs the GPU)
  python scripts/planted_bugs.py gpu-abdada     # ABDADA's kernel and driver (build/mut-build; needs the GPU)
  python scripts/planted_bugs.py gui-look       # the GUI's look and feel (pytest in build/mut-gui)
  python scripts/planted_bugs.py gui-analysis   # the GUI's game tree, PGN, analysis and reviews
  python scripts/planted_bugs.py gui-events     # the GUI's tournaments and databases
  python scripts/planted_bugs.py <campaign> --only TEXT   # only the bugs whose title contains TEXT (repeatable)

Every campaign plants its bugs in a copy of src/, tests/ and the CMake files
and builds that copy in a tree of its own, so the working tree stays free
(edit and build there meanwhile) and keeps its timestamps: CPU campaigns use
build/mut-src-cpu and build/mut-build-cpu (target unit_tests, -j 8, safe
while a match plays); GPU campaigns use build/mut-src and build/mut-build (a
reduced tree: quad board, pesto, staged ordering, the killer and SEE feature
sets, wide128) and need the GPU to themselves; GUI campaigns copy gui/ into
build/mut-gui and run pytest there. 'snapshot' copies the sources for a later
GPU run with --keep-snapshot. Each campaign logs to
build/planted_bugs/<campaign>.log. Never run two of the same kind at once.
"""

import argparse
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "build" / "planted_bugs"
MUT_SRC = ROOT / "build" / "mut-src"      # the GPU campaigns' copy of the sources
MUT_BUILD = ROOT / "build" / "mut-build"
MUT_CPU_SRC = ROOT / "build" / "mut-src-cpu"  # the CPU campaigns' copy
MUT_CPU_BUILD = ROOT / "build" / "mut-build-cpu"
MUT_GUI = ROOT / "build" / "mut-gui"  # the GUI campaigns' copy (pytest runs there)

MO = "src/core/move_order.h"
SE = "src/core/search.h"
SEE = "src/core/see.h"
YK = "src/gpu/ybwc_kernels.cuh"
YD = "src/gpu/ybwc_device.cu"
SK = "src/gpu/search_kernels.cuh"

# (name, file, text, replacement)
KILLERS_CPU = [
    ("staged killer equal to the PV returned again", MO,
     "if (!k.is_none() && can_be_killer(k) && k != pv &&",
     "if (!k.is_none() && can_be_killer(k) &&"),
    ("staged killers not checked for legality", MO,
     "(st.aux == 1 || k != Move::from_raw(st.killers[0])) && pv_is_legal(b, a, k))",
     "(st.aux == 1 || k != Move::from_raw(st.killers[0])))"),
    ("staged duplicate killer returned twice", MO,
     "(st.aux == 1 || k != Move::from_raw(st.killers[0])) && pv_is_legal(b, a, k))",
     "pv_is_legal(b, a, k))"),
    ("staged killers in the wrong order", MO,
     "const Move k = Move::from_raw(st.killers[st.aux]);",
     "const Move k = Move::from_raw(st.killers[1 - st.aux]);"),
    ("staged quiet stage doesn't skip killers", MO,
     "if (m != pv && m.raw() != st.killers[0] && m.raw() != st.killers[1] &&",
     "if (m != pv &&"),
    ("staged killers of other kinds returned", MO,
     "if (!k.is_none() && can_be_killer(k) && k != pv &&",
     "if (!k.is_none() && k != pv &&"),
    ("order key: killers before castling and queen promotions", MO,
     "    if (flags == PROMO_QUEEN) return (order::kQueenPromotion << 29) | ((63 - from) << 20);\n",
     "    if (killers.has(m)) return (order::kKiller << 29) | (m == killers.first ? (1u << 28) : 0u);\n"
     "    if (flags == PROMO_QUEEN) return (order::kQueenPromotion << 29) | ((63 - from) << 20);\n"),
    ("order key: the older killer first", MO,
     "if (killers.has(m)) return (order::kKiller << 29) | (m == killers.first ? (1u << 28) : 0u);",
     "if (killers.has(m)) return (order::kKiller << 29) | (m == killers.second ? (1u << 28) : 0u);"),
    ("add_killer shifts even when the move is the newer killer", MO,
     "return k.first == m ? k : Killers{m, k.first};",
     "return Killers{m, k.first};"),
    ("list picker ignores killers", MO,
     "mode == PickMode::All ? killers : Killers{});",
     "Killers{});"),
    ("search never orders with killers", SE,
     "if constexpr (Features::kKillers) node_killers = killers[ply];",
     "if constexpr (Features::kKillers) node_killers = Killers{};"),
    ("search reads the killers of the ply before", SE,
     "if constexpr (Features::kKillers) node_killers = killers[ply];",
     "if constexpr (Features::kKillers) node_killers = killers[ply > 0 ? ply - 1 : 0];"),
    ("search stores at the child's ply", SE,
     "if (can_be_killer(cur.move)) killers[ply] = add_killer(killers[ply], cur.move);",
     "if (can_be_killer(cur.move)) killers[ply + 1] = add_killer(killers[ply + 1], cur.move);"),
    ("search stores captures too", SE,
     "if (can_be_killer(cur.move)) killers[ply] = add_killer(killers[ply], cur.move);",
     "killers[ply] = add_killer(killers[ply], cur.move);"),
    ("search stores at every new best move, not only cutoffs", SE,
     "                if (v > cur.alpha) {\n                    cur.alpha = short(v);\n",
     "                if constexpr (Features::kKillers) {\n"
     "                    if (can_be_killer(cur.move)) killers[ply] = add_killer(killers[ply], cur.move);\n"
     "                }\n"
     "                if (v > cur.alpha) {\n                    cur.alpha = short(v);\n"),
]

SEE_CPU = [
    ("least valuable attacker: knights before pawns", SEE,
     """        u64 pick = mine & b.template pieces<PAWN>();
        int pt = PAWN;
        if (!pick) {
            pick = mine & b.template pieces<KNIGHT>();
            pt = KNIGHT;
        }""",
     """        u64 pick = mine & b.template pieces<KNIGHT>();
        int pt = KNIGHT;
        if (!pick) {
            pick = mine & b.template pieces<PAWN>();
            pt = PAWN;
        }"""),
    ("no x-rays", SEE,
     """        attackers = (attackers | (a.bishop(to, occupied) & b.diagonal_sliders()) |
                     (a.rook(to, occupied) & b.straight_sliders())) &
                    occupied;""",
     """        attackers &= occupied;"""),
    ("no king rule", SEE,
     """        if (pt == KING && (attackers_to(b, a, to, occupied) & occupied & b.color(~side))) break;""",
     ""),
    ("king rule ignores sliders behind the king", SEE,
     """        if (pt == KING && (attackers_to(b, a, to, occupied) & occupied & b.color(~side))) break;""",
     """        if (pt == KING && (attackers & occupied & b.color(~side))) break;"""),
    ("back-up step without the negation", SEE,
     """    for (; d > 0; --d) gain[d - 1] = -(-gain[d - 1] > gain[d] ? -gain[d - 1] : gain[d]);""",
     """    for (; d > 0; --d) gain[d - 1] = (gain[d - 1] > gain[d] ? gain[d - 1] : gain[d]);"""),
    ("en passant victim stays on the board", SEE,
     """    if (m.flags() == EP_CAPTURE) occupied ^= bit(Square(int(to) - forward_offset(us)));
""",
     ""),
    ("promoted piece counted as a pawn on the square", SEE,
     """    int on_square = m.is_promotion() ? see_value(m.promotion_type()) : see_value(b.piece_type_on(from));""",
     """    int on_square = see_value(b.piece_type_on(from));"""),
    ("en passant gains nothing", SEE,
     """    int gain = m.flags() == EP_CAPTURE ? see_value(PAWN) : (m.is_capture() ? see_value(b.piece_type_on(m.to())) : 0);""",
     """    int gain = m.is_capture() ? see_value(b.piece_type_on(m.to())) : 0;"""),
    ("delta pruning only below alpha", SEE,
     """        if (bound <= alpha) return bound;""",
     """        if (bound < alpha) return bound;"""),
    ("SEE pruning also prunes even trades", SEE,
     """    if (prune.see_prune && see(b, a, m) < 0) return -kScoreInfinite;""",
     """    if (prune.see_prune && see(b, a, m) <= 0) return -kScoreInfinite;"""),
    ("delta margin ignored", SEE,
     """        const int bound = stand + material_gain(b, m) + prune.delta_margin;""",
     """        const int bound = stand + material_gain(b, m);"""),
    ("delta pruning leaves best alone", SE,
     """                        if (bound > cur.best) cur.best = short(bound);  // delta: at least the bound (fail-soft)""",
     """                        // (best left alone)"""),
    ("pruning in check too", SE,
     """                    (cur.flags & (level_flags::kQuiescence | level_flags::kInCheck)) == level_flags::kQuiescence) {""",
     """                    (cur.flags & level_flags::kQuiescence) != 0) {"""),
    ("pruning in the main search too", SE,
     """                    (cur.flags & (level_flags::kQuiescence | level_flags::kInCheck)) == level_flags::kQuiescence) {""",
     """                    (cur.flags & level_flags::kInCheck) == 0) {"""),
    ("stand pat taken as the best so far", SE,
     """                    const int bound = quiescence_prune_bound(sb.board, a, env.prune, m, stand, cur.alpha);""",
     """                    const int bound = quiescence_prune_bound(sb.board, a, env.prune, m, cur.best, cur.alpha);"""),
]

# (name, file, text, replacement, test filter)
GPU_EXPLICIT = [
    # name, file, old, new, test filter
    ("explicit nodes never store killers", YK,
     """            if (m.is_none() || !can_be_killer(m)) return;""",
     """            if (true) return;""", "gpu_killers"),
    ("explicit nodes order without killers", YK,
     """            const u32 k = order_key(b, list.moves[i], first, killers);""",
     """            const u32 k = order_key(b, list.moves[i], first);""", "gpu_killers"),
    ("killers stored at the child's ply", YK,
     """                store_killer(par.ply, Move::from_raw(ch.move));
                out = par.best;""",
     """                store_killer(par.ply + 1, Move::from_raw(ch.move));
                out = par.best;""", "gpu_killers"),
    ("killers read from the next ply", YK,
     """        const Killers killers = load_killers(ply);""",
     """        const Killers killers = load_killers(ply + 1);""", "gpu_killers"),
    ("the parent's own move stored as killer", YK,
     """                store_killer(par.ply, Move::from_raw(ch.move));
                out = par.best;""",
     """                store_killer(par.ply, Move::from_raw(par.move));
                out = par.best;""", "gpu_killers"),
    ("killer table not emptied before a search", YD,
     """        if (d_killers) CUDA_CHECK(cudaMemsetAsync(d_killers, 0, (kMaxPly + 1) * sizeof(u32), work));""",
     "", "gpu_killers"),
    ("warp list ordering ignores killers", SK,
     """            const u32 key = order_key(b, mine.moves[i], mode == PickMode::All ? pv : Move(),
                                      mode == PickMode::All ? killers : Killers{});""",
     """            const u32 key = order_key(b, mine.moves[i], mode == PickMode::All ? pv : Move(),
                                      Killers{});""", "gpu_killers"),
    ("explicit quiescence nodes never prune", YK,
     """                if (stand_pat && (list.moves[i].is_capture() || list.moves[i].is_promotion())) {
                    const int bound = quiescence_prune_bound(b, attacks, a.prune, list.moves[i], best, alpha);""",
     """                if (false) {
                    const int bound = quiescence_prune_bound(b, attacks, a.prune, list.moves[i], best, alpha);""",
     "gpu_see"),
    ("explicit quiescence pruning leaves best alone", YK,
     """            if (qs_bound > best) best = qs_bound;  // at least the bound (fail-soft), as in search_task""",
     """            // (best left alone)""", "gpu_see"),
    ("explicit quiescence nodes prune with the parent's alpha", YK,
     """                    const int bound = quiescence_prune_bound(b, attacks, a.prune, list.moves[i], best, alpha);""",
     """                    const int bound = quiescence_prune_bound(b, attacks, a.prune, list.moves[i], best, x.alpha - 50);""",
     "gpu_see"),
]


AB = "src/core/abdada.h"
AK = "src/gpu/abdada_kernels.cuh"
AD = "src/gpu/abdada_device.cu"

# ABDADA's shared code (core/abdada.h, the list picker's deferral, the hooks
# in search_task), CPU side: tests/test_abdada.cpp.
ABDADA_CPU = [
    ("own claims count as busy", AB,
     "return owner != 0 && owner != u64(worker) + 1 && (slot & kKeyMask) == (key & kKeyMask);",
     "return owner != 0 && (slot & kKeyMask) == (key & kKeyMask);"),
    ("claims lose the worker", AB,
     "return (key & kKeyMask) | (u64(worker) + 1);",
     "return (key & kKeyMask) | 1;"),
    ("busy ignores the key", AB,
     "return owner != 0 && owner != u64(worker) + 1 && (slot & kKeyMask) == (key & kKeyMask);",
     "return owner != 0 && owner != u64(worker) + 1;"),
    ("a deferred move is lost", MO,
     """        --st.picked;
        arena.keys[st.begin + st.picked] = kDeferredKey;""",
     """        arena.keys[st.begin + st.picked - 1] = kDeferredKey;"""),
    ("a deferred move comes back first", MO,
     "        arena.keys[st.begin + st.picked] = kDeferredKey;",
     "        arena.keys[st.begin + st.picked] = 0xFFFFFFFFu;"),
    ("returned deferred moves not recognized", MO,
     "return st.picked > 0 && arena.keys[st.begin + st.picked - 1] == kDeferredKey;",
     "return false;"),
    ("deferral without unmaking the move", SE,
     """                            unmake_search_move(sb, m, cur.undo, cur.key, tables);
                            Ordering::defer(cur.picker, ws);""",
     """                            Ordering::defer(cur.picker, ws);"""),
    ("claims never released", SE,
     "if (!from_null && cur.depth >= control.params.min_depth) control.release(sb.key);",
     "(void)from_null;"),
    ("claims released one ply short", SE,
     "if (!from_null && cur.depth >= control.params.min_depth) control.release(sb.key);",
     "if (!from_null && cur.depth > control.params.min_depth) control.release(sb.key);"),
    ("the parent claimed instead of the child", SE,
     "if (cur.depth >= control.params.min_depth) control.claim(sb.key);",
     "if (cur.depth >= control.params.min_depth) control.claim(cur.key);"),
    ("deferrals not counted", SE,
     """                            Ordering::defer(cur.picker, ws);
                            control.count_deferred();""",
     """                            Ordering::defer(cur.picker, ws);"""),
]

# ABDADA's GPU-only code (the kernel and its driver): tests/gpu_abdada_tests.cu.
GPU_ABDADA = [
    ("device claims never stored", AK,
     "        *static_cast<volatile u64*>(&busy_table[key & mask]) = abdada::slot_value(key, worker);",
     "        (void)key;"),
    ("device claims stored for another worker", AK,
     "= abdada::slot_value(key, worker);",
     "= abdada::slot_value(key, worker + 1);"),
    ("device claims never released", AK,
     """        atomicCAS(reinterpret_cast<unsigned long long*>(&busy_table[key & mask]),
                  (unsigned long long)abdada::slot_value(key, worker), 0ull);""",
     """        (void)key;"""),
    ("device busy reads another slot", AK,
     "const u64 slot = *static_cast<volatile const u64*>(&busy_table[key & mask]);",
     "const u64 slot = *static_cast<volatile const u64*>(&busy_table[(key >> 1) & mask]);"),
    ("the first finisher doesn't stop the others", AK,
     "return *static_cast<volatile const int*>(done) != 0 || stop_requested(signal);",
     "return stop_requested(signal);"),
    ("worker numbers ignore the lanes", AK,
     "const unsigned worker = warp * unsigned(a.lanes) + lane;",
     "const unsigned worker = warp + lane;"),
    ("node counts not summed", AK,
     "    atomicAdd(a.nodes, (unsigned long long)r.nodes);",
     ""),
    ("one block too few", AD,
     "const int blocks = int((size_t(warps) * 32 + size_t(options.block_size) - 1) / size_t(options.block_size));",
     "const int blocks = std::max(1, int((size_t(warps) * 32) / size_t(options.block_size)));"),
    ("game history not passed", AD,
     "a.history = game_count > 0 ? d_history : nullptr;",
     "a.history = nullptr;"),
    ("PV hint not passed", AD,
     "a.pv = pv_length > 0 ? d_pv : nullptr;",
     "a.pv = nullptr;"),
    ("busy table not cleared", AD,
     "        CUDA_CHECK(cudaMemsetAsync(d_busy, 0, busy_slots * sizeof(u64), work));",
     ""),
    ("the root window ignored", AD,
     "        h_task_.alpha = short(alpha);",
     "        h_task_.alpha = short(-kScoreInfinite);"),
    ("the table not passed", AD,
     "        a.tt = tt;",
     ""),
]


M32 = "src/core/move32.h"

# 32-bit moves (core/move32.h), CPU side: tests/test_move32.cpp.
MOVE32_CPU = [
    ("moving piece stored a bit off", M32,
     "        : v_(u32(m.raw()) | (u32(moving) << 16) | (u32(captured) << 20)) {}",
     "        : v_(u32(m.raw()) | (u32(moving) << 17) | (u32(captured) << 20)) {}"),
    ("en passant records no victim", M32,
     "            m.flags() == EP_CAPTURE ? PAWN : m.is_capture() ? board->piece_type_on(m.to()) : NO_PIECE_TYPE;",
     "            m.is_capture() ? board->piece_type_on(m.to()) : NO_PIECE_TYPE;"),
    ("a piece's moves take the first target's piece as the mover", M32,
     "        const PieceType moving = board->piece_type_on(from);",
     "        const PieceType moving = targets ? board->piece_type_on(Square(lsb(targets))) : NO_PIECE_TYPE;"),
    ("pawn captures record no victim", M32,
     "                                   flags == CAPTURE ? board->piece_type_on(to) : NO_PIECE_TYPE);",
     "                                   NO_PIECE_TYPE);"),
    ("promotion captures record no victim", M32,
     "            const PieceType captured = capture ? board->piece_type_on(to) : NO_PIECE_TYPE;",
     "            const PieceType captured = NO_PIECE_TYPE;"),
    ("unmake moves the captured piece back", M32,
     "    detail::unmake_move_known(b, m.move(), m.moving(), undo, obs);",
     "    detail::unmake_move_known(b, m.move(), m.captured() != NO_PIECE_TYPE ? m.captured() : m.moving(), undo, obs);"),
    ("bind keeps the old moves", M32,
     """        board = &b;
        size = 0;""",
     """        board = &b;"""),
]


NN = "src/core/nnue.h"
NT = "src/host/nnue_train.cpp"

# NNUE (core/nnue.h, host/nnue_train.cpp), CPU side: tests/test_nnue.cpp.
NNUE_CPU = [
    ("black's view not mirrored", NN,
     "    const int sq = perspective == WHITE ? int(s) : (int(s) ^ 56);",
     "    const int sq = int(s);"),
    ("add updates white's view only", NN,
     """            acc.v[WHITE][i] = i16(acc.v[WHITE][i] + load_readonly(&w[i]));
            acc.v[BLACK][i] = i16(acc.v[BLACK][i] + load_readonly(&b[i]));""",
     """            acc.v[WHITE][i] = i16(acc.v[WHITE][i] + load_readonly(&w[i]));"""),
    ("remove adds", NN,
     """            acc.v[WHITE][i] = i16(acc.v[WHITE][i] - load_readonly(&w[i]));""",
     """            acc.v[WHITE][i] = i16(acc.v[WHITE][i] + load_readonly(&w[i]));"""),
    ("both halves use the side to move's output weights", NN,
     "            out += nnue_clip(acc.v[~side_to_move][i] + bias) * int(load_readonly(&n.w2[1][i]));",
     "            out += nnue_clip(acc.v[~side_to_move][i] + bias) * int(load_readonly(&n.w2[0][i]));"),
    ("activations clip at twice the scale", NN,
     "CB_HD constexpr int nnue_clip(int x) { return x < 0 ? 0 : (x > kNnueQA ? kNnueQA : x); }",
     "CB_HD constexpr int nnue_clip(int x) { return x < 0 ? 0 : (x > 2 * kNnueQA ? 2 * kNnueQA : x); }"),
    ("the bias left out of the other side's half", NN,
     "            out += nnue_clip(acc.v[~side_to_move][i] + bias) * int(load_readonly(&n.w2[1][i]));",
     "            out += nnue_clip(acc.v[~side_to_move][i]) * int(load_readonly(&n.w2[1][i]));"),
    ("output layer quantized at the first layer's scale", NT,
     "        w.w2[0][i] = i16(q(n.w2[size_t(i)], kNnueQB, kMaxW2));",
     "        w.w2[0][i] = i16(q(n.w2[size_t(i)], kNnueQA, kMaxW2));"),
    ("gradient of the wrong sign", NT,
     "    const float dy = 2.0f * err * p * (1.0f - p) * kLn10;",
     "    const float dy = -2.0f * err * p * (1.0f - p) * kLn10;"),
    ("black's gradient goes to white's inputs", NT,
     "        float* b = &g.w1[size_t(s.black[j]) * size_t(h)];",
     "        float* b = &g.w1[size_t(s.white[j]) * size_t(h)];"),
    # Other widths (train-nnue --hidden): tests/test_nnue.cpp, nnue_other_widths_train.
    ("the float network's output reads the search's width", NT,
     "    for (int i = 0; i < hidden; ++i) y += w2[size_t(i)] * clip01(us[i]) + w2[size_t(hidden + i)] * clip01(them[i]);",
     "    for (int i = 0; i < hidden; ++i) y += w2[size_t(i)] * clip01(us[i]) + w2[size_t(kNnueHidden + i)] * clip01(them[i]);"),
    ("the gradient keeps the default width", NT,
     "    out.hidden = net.hidden;\n",
     ""),
    # Weight decay (train-nnue --weight-decay): nnue_weight_decay_shrinks_the_weights.
    ("decay on the first layer's biases too", NT,
     "            adam_b1.step(net.b1.data(), total.b1.data(), net.b1.size(), lr, step, kMaxB1);",
     "            adam_b1.step(net.b1.data(), total.b1.data(), net.b1.size(), lr, step, kMaxB1, o.weight_decay);"),
    ("no decay on the output layer", NT,
     "            adam_w2.step(net.w2.data(), total.w2.data(), net.w2.size(), lr, step, kMaxW2, o.weight_decay);",
     "            adam_w2.step(net.w2.data(), total.w2.data(), net.w2.size(), lr, step, kMaxW2);"),
    ("decay without the step size", NT,
     "        const float keep = 1.0f - lr * decay;",
     "        const float keep = 1.0f - decay;"),
    ("saved without the output bias", NT,
     "    out.write(reinterpret_cast<const char*>(&w), sizeof w);",
     "    out.write(reinterpret_cast<const char*>(&w), sizeof w - sizeof w.b2);"),
]


# Principal variation search (core/search.h), CPU side: tests/test_pvs.cpp.
PVS_CPU = [
    ("the first move gets the zero window too", SE,
     "zero_window = cur.depth > 0 && cur.moves_searched > 1 && cur.beta - cur.alpha > 1;",
     "zero_window = cur.depth > 0 && cur.moves_searched > 0 && cur.beta - cur.alpha > 1;"),
    ("zero windows in quiescence too", SE,
     "zero_window = cur.depth > 0 && cur.moves_searched > 1 && cur.beta - cur.alpha > 1;",
     "zero_window = cur.depth >= 0 && cur.moves_searched > 1 && cur.beta - cur.alpha > 1;"),
    ("zero windows at zero-window nodes too", SE,
     "zero_window = cur.depth > 0 && cur.moves_searched > 1 && cur.beta - cur.alpha > 1;",
     "zero_window = cur.depth > 0 && cur.moves_searched > 1;"),
    ("a zero-window fail high is never searched again", SE,
     """            if constexpr (Features::kLmr || pvs_v<Features>) {
                reduced = (cur.flags & level_flags::kReduced) != 0;""",
     """            if constexpr (Features::kLmr) {
                reduced = (cur.flags & level_flags::kReduced) != 0;"""),
    ("the window is narrowed on the wrong side", SE,
     """                        if (zero_window && !(cur.flags & level_flags::kReduced)) {
                            cur.flags |= level_flags::kReduced;
                            child_alpha = short(-cur.alpha - 1);""",
     """                        if (zero_window && !(cur.flags & level_flags::kReduced)) {
                            cur.flags |= level_flags::kReduced;
                            child_alpha = short(-cur.beta + 1);"""),
]


# Checks in quiescence (core/move_order.h, core/search.h), CPU side:
# tests/test_qchecks.cpp.
QCHECKS_CPU = [
    ("staged: quiet checks after the killers and castling", MO,
     """                    if (st.mode == u8(PickMode::QuiescenceChecks) && !ci.in_check()) {
                        // Straight to the quiet moves, which only checks pass.
                        st.stage = kStageQuiets;
                        start_quiet_kind(st, b, a, ci, order::kQuietKnight);
                        break;
                    }""",
     """"""),
    ("staged: every quiet move, checks or not", MO,
     """                            (st.mode != u8(PickMode::QuiescenceChecks) || quiet_move_gives_check(b, m, a)))""",
     """                            (st.mode != u8(PickMode::QuiescenceChecks) || true))"""),
    ("staged: under-promotions after the quiet checks", MO,
     """                    if (st.mode == u8(PickMode::QuiescenceChecks)) {
                        st.stage = kStageDone;  // no under-promotions in quiescence
                        break;
                    }""",
     """"""),
    ("staged: killers kept outside the main search", MO,
     """        st.killers[0] = mode == PickMode::All ? killers.first.raw() : u16(0);""",
     """        st.killers[0] = killers.first.raw();"""),
    ("list: castling counted as a quiet check", MO,
     """    return m.flags() <= DOUBLE_PUSH && quiet_move_gives_check(b, m, a);""",
     """    return m.flags() <= QUEEN_CASTLE && gives_check(b, m, a);"""),
    # The targeted generation of the staged picker's quiet checks.
    ("staged: blockers of our sliders get only direct targets", MO,
     """        if (a.line(king, from) && slider_sees_king(b, a, ci.us, king, from, occupied ^ bit(from))) return ~0ull;""",
     """        if (false && a.line(king, from)) return ~0ull;"""),
    ("staged: blockers looked for on our own king's lines", MO,
     """        if (a.line(king, from) && slider_sees_king(b, a, ci.us, king, from, occupied ^ bit(from))) return ~0ull;""",
     """        if (a.line(ci.king, from) && slider_sees_king(b, a, ci.us, king, from, occupied ^ bit(from))) return ~0ull;"""),
    ("staged: blockers tested with the blocker still there", MO,
     """        if (a.line(king, from) && slider_sees_king(b, a, ci.us, king, from, occupied ^ bit(from))) return ~0ull;""",
     """        if (a.line(king, from) && slider_sees_king(b, a, ci.us, king, from, occupied)) return ~0ull;"""),
    ("staged: knight checks from the king's squares", MO,
     """            case order::kQuietKnight: return a.knight(king);""",
     """            case order::kQuietKnight: return a.king(king);"""),
    ("staged: bishop checks along files and ranks", MO,
     """            case order::kQuietBishop: return a.bishop(king, occupied);""",
     """            case order::kQuietBishop: return a.rook(king, occupied);"""),
    ("staged: queen checks along diagonals only", MO,
     """            case order::kQuietQueen: return a.queen(king, occupied);""",
     """            case order::kQuietQueen: return a.bishop(king, occupied);"""),
    ("staged: pawn blockers ignored", MO,
     """                    if (slider_sees_king(b, a, us, king, from, b.occupied() ^ bit(from))) blockers |= bit(from);""",
     """                    if (false && slider_sees_king(b, a, us, king, from, b.occupied() ^ bit(from))) blockers |= bit(from);"""),
    ("staged: double pushes of blockers one rank short", MO,
     """                                                                  : shift_forward(shift_forward(blockers, us), us);""",
     """                                                                  : shift_forward(blockers, us);"""),
    ("staged: pawn checks from our own pawn's attacks", MO,
     """                checks = a.pawn(~us, king) | moved;""",
     """                checks = a.pawn(us, king) | moved;"""),
    ("search: quiet checks at every quiescence ply", SE,
     """                            if (mode == PickMode::Quiescence && cur.depth == 0) mode = PickMode::QuiescenceChecks;""",
     """                            if (mode == PickMode::Quiescence && cur.depth >= -1) mode = PickMode::QuiescenceChecks;"""),
    ("search: quiet checks pruned like captures", SE,
     """                if (!null_move && !m.is_none() && (m.is_capture() || m.is_promotion()) &&""",
     """                if (!null_move && !m.is_none() &&"""),
]


# Aspiration windows and the fail-low extension (host/search_backend.cpp),
# CPU side: tests/test_aspiration.cpp (aspiration_*, plan_time_*).
SB = "src/host/search_backend.cpp"
ASPIRATION_CPU = [
    ("a fail low keeps the hard limit", SB,
     "                    if (request.fail_low > request.hard) backend.set_deadline(request.fail_low);\n",
     ""),
    ("the extension outlives its iteration", SB,
     "            backend.set_deadline(request.hard);",
     "            if (depth == 2) backend.set_deadline(request.hard);"),
    ("fail highs extended too", SB,
     "                    beta = is_mate_score(s) ? kScoreInfinite : std::min(kScoreInfinite, s + delta);",
     "                    beta = is_mate_score(s) ? kScoreInfinite : std::min(kScoreInfinite, s + delta);\n"
     "                    if (request.fail_low > request.hard) backend.set_deadline(request.fail_low);"),
    ("the extension not capped by the clock", SB,
     "std::min(b.hard_ms * (1.0 + fail_low_percent / 100.0), usable * 0.4)",
     "b.hard_ms * (1.0 + fail_low_percent / 100.0)"),
    ("a fixed move time extended", SB,
     "        b.soft_ms = b.fail_low_ms = b.hard_ms;",
     "        b.soft_ms = b.hard_ms;\n        b.fail_low_ms = b.hard_ms * (1.0 + fail_low_percent / 100.0);"),
    ("the margin never doubles", SB,
     "                delta *= 2;",
     "                delta *= 1;"),
    ("the window centred on the wrong side", SB,
     "            int beta = std::min(kScoreInfinite, out.score + delta);",
     "            int beta = std::min(kScoreInfinite, out.score + 2 * delta);"),
]


# The GUI's look and feel (gui/chessgpu_gui: theme, sounds, board, panels,
# dialogs, main window): gui/tests/test_look.py, test_main_window.py and
# test_widgets.py, run with pytest in a copy of gui/ (no build needed).
G = "gui/chessgpu_gui/"
GUI_LOOK_TESTS = ["tests/test_look.py", "tests/test_main_window.py", "tests/test_widgets.py"]
GUI_LOOK = [
    ("theme: the palette isn't applied", G + "theme.py",
     "    app.setPalette(palette(_current))\n", ""),
    ("theme: the current theme isn't recorded", G + "theme.py",
     "    _current = THEMES.get(name, THEMES[DEFAULT_THEME])\n    app.setStyle",
     "    THEMES.get(name, THEMES[DEFAULT_THEME])\n    app.setStyle"),
    ("theme: an unknown name fails instead of falling back", G + "theme.py",
     "    _current = THEMES.get(name, THEMES[DEFAULT_THEME])",
     "    _current = THEMES[name]"),
    ("theme: the light theme keeps the dark window color", G + "theme.py",
     'window="#eef0f3"', 'window="#1e1f22"'),
    ("theme: the palette's window is the surface color", G + "theme.py",
     "QPalette.ColorRole.Window: t.window,", "QPalette.ColorRole.Window: t.surface,"),
    ("sounds: noise bursts click in", G + "sounds.py",
     "rng.uniform(-1.0, 1.0) * min(1.0, i / ramp) * math.exp", "rng.uniform(-1.0, 1.0) * math.exp"),
    ("sounds: volume ignored", G + "sounds.py",
     'struct.pack("<h", int(round(32767 * volume * v)))', 'struct.pack("<h", int(round(32767 * v)))'),
    ("sounds: promotion outranks check", G + "sounds.py",
     '    if after.in_check():\n        return "check"\n    if before.is_castling(move):',
     '    if move.promotion:\n        return "promote"\n    if after.in_check():\n        return "check"\n'
     '    if before.is_castling(move):'),
    ("sounds: en passant sounds like a quiet move", G + "sounds.py",
     "    if before.is_capture(move):", '    if before.board[move.to_sq] != ".":'),
    ("sounds: one file for every volume", G + "sounds.py",
     'path = self.folder / f"{name}-{step}.wav"', 'path = self.folder / f"{name}.wav"'),
    ("sounds: plays while disabled", G + "sounds.py",
     "        if not self.enabled or self.backend is None", "        if self.backend is None"),
    ("sounds: a missing audio device raises", G + "sounds.py",
     "        except (OSError, RuntimeError):", "        except RuntimeError:"),
    ("board: castling (four squares) isn't animated", G + "board_view.py",
     "    if not 2 <= len(changed) <= 4:", "    if not 2 <= len(changed) <= 3:"),
    ("board: captured pieces don't fade", G + "board_view.py",
     "    plan.fade_out = sorted((p, sq) for sq, p in vanished.items())", "    plan.fade_out = []"),
    ("board: promotions aren't matched to their pawn", G + "board_view.py",
     "    match(False)  # a pawn promoting", "    # match(False)  # a pawn promoting"),
    ("board: the animation never starts", G + "board_view.py",
     "                self._animation.start()\n", ""),
    ("board: Shift draws blue", G + "board_view.py",
     '        if modifiers & Qt.KeyboardModifier.ShiftModifier:\n            return "red"',
     '        if modifiers & Qt.KeyboardModifier.ShiftModifier:\n            return "blue"'),
    ("board: a left click keeps the shapes", G + "board_view.py",
     "        self.clear_shapes()\n        if sq is None or not (self.interactive", "        if sq is None or not (self.interactive"),
    ("board: premove targets for the side to move", G + "board_view.py",
     "        mover = Position(pos.board, self.premove_color, pos.castling, None, pos.halfmove, pos.fullmove)",
     "        mover = Position(pos.board, pos.turn, pos.castling, None, pos.halfmove, pos.fullmove)"),
    ("board: premoved promotions don't queen", G + "board_view.py",
     '        promotion = "q" if piece in "Pp" and rank_of(to_sq) == last_rank else ""', '        promotion = ""'),
    ("board: a right click keeps the premove", G + "board_view.py",
     "            if had_premove:\n                self.cancel_premoves()\n", ""),
    ("board: auto-queen still asks", G + "board_view.py",
     '            choice = "q" if self.auto_queen else self.promotion_chooser(self.position.turn, to_sq)',
     "            choice = self.promotion_chooser(self.position.turn, to_sq)"),
    ("board: the overlay lists the rook second", G + "board_view.py",
     'PROMOTION_ORDER = "qnrb"', 'PROMOTION_ORDER = "qrnb"'),
    ("board: the overlay always opens downward", G + "board_view.py",
     "        down = rect.top() < self.board_rect().center().y()", "        down = True"),
    ("board: hover marks any square", G + "board_view.py",
     "            self._hover = sq if sq is not None and sq in self.targets(self._drag_from) else None",
     "            self._hover = sq"),
    ("board: the board theme is ignored", G + "board_view.py",
     "        light, dark = QColor(self.board_theme.light), QColor(self.board_theme.dark)",
     '        light, dark = QColor("#f0d9b5"), QColor("#b58863")'),
    ("panels: promotions keep their letter", G + "panels.py",
     '    if "=" in san:', '    if False:'),
    ("panels: material lead from Black's side", G + "panels.py",
     "    lead = sum(PIECE_VALUES[k] * (counts[WHITE][k] - counts[BLACK][k]) for k in PIECE_VALUES)",
     "    lead = sum(PIECE_VALUES[k] * (counts[BLACK][k] - counts[WHITE][k]) for k in PIECE_VALUES)"),
    ("panels: extra pieces credited to the other side", G + "panels.py",
     "        side = WHITE if diff > 0 else BLACK", "        side = BLACK if diff > 0 else WHITE"),
    ("panels: cards only as wide as the board", G + "panels.py",
     "        total_w = bar + gap + side", "        total_w = side"),
    ("panels: the board area's width ignores the eval bar", G + "panels.py",
     "        ideal = 2 * margin + self.eval_bar.width() + gap + side", "        ideal = 2 * margin + gap + side"),
    ("panels: low time from the other end", G + "panels.py",
     '        return "low" if g.tc.clocked and g.clock.left(self.color) < LOW_TIME_MS else "active"',
     '        return "low" if g.tc.clocked and g.clock.left(self.color) > LOW_TIME_MS else "active"'),
    ("panels: untimed games show a clock", G + "panels.py",
     "        self.clock.setVisible(clocked)", "        self.clock.setVisible(True)"),
    ("panels: every navigation button goes forward", G + "panels.py",
     "            b.clicked.connect(getattr(self, name).emit)", "            b.clicked.connect(self.next.emit)"),
    ("panels: the move list ignores figurines", G + "panels.py",
     "                    text = figurine_san(m.san) if self.figurines else m.san", "                    text = m.san"),
    ("settings: any stored text reads as true", G + "dialogs.py",
     '                    value = raw if isinstance(raw, bool) else str(raw).lower() == "true"',
     "                    value = raw if isinstance(raw, bool) else bool(raw)"),
    ("settings: volume not clamped", G + "dialogs.py",
     "        out.volume = max(0, min(100, out.volume))\n", ""),
    ("settings: an unknown theme is kept", G + "dialogs.py",
     "        if out.theme not in THEMES:\n            out.theme = DEFAULT_THEME\n", ""),
    ("settings: the dialog drops figurines", G + "dialogs.py",
     "            figurines=self.figurines.isChecked())", "            figurines=False)"),
    ("settings: the volume stays enabled with sounds off", G + "dialogs.py",
     "        self.sounds.toggled.connect(self.volume.setEnabled)\n", ""),
    ("window: no sound at the start of a game", G + "main_window.py",
     '            self.sounds.play("start")\n', ""),
    ("window: jumps through a game play sounds", G + "main_window.py",
     "        step = node.parent is self.view_node", "        step = node is not self.view_node"),
    ("window: the low-time tick repeats", G + "main_window.py",
     "            self._warned.add(human)\n", ""),
    ("window: queued premoves are never played", G + "main_window.py",
     "            QTimer.singleShot(0, lambda m=move: g.human_move(m))\n", "            pass\n"),
    ("window: premoves allowed with the option off", G + "main_window.py",
     "        self.board.set_premove_color(human if waiting and self.view.premoves else None)",
     "        self.board.set_premove_color(human if waiting else None)"),
    ("window: the cards stay put on flip", G + "main_window.py",
     "        self.eval_bar.set_flipped(self.board.flipped)\n        self._place_cards()\n",
     "        self.eval_bar.set_flipped(self.board.flipped)\n"),
    ("window: settings aren't saved", G + "main_window.py",
     "        v.save(self.settings)\n", ""),
    ("window: a new theme isn't applied", G + "main_window.py",
     "            if app is not None:\n                apply_theme(app, v.theme)\n            self._retheme_icons()",
     "            self._retheme_icons()"),
]


def cpu_configure():
    snapshot_sources(MUT_CPU_SRC)
    subprocess.run(["cmake", "-S", str(MUT_CPU_SRC), "-B", str(MUT_CPU_BUILD), "-DCHESSGPU_BOARDS=quad",
                    "-DCHESSGPU_MODELS=thread", "-DCHESSGPU_EVALS=pesto", "-DCHESSGPU_FEATURES=none",
                    "-DCHESSGPU_TT_LAYOUTS=wide128"], check=True, capture_output=True)


def cpu_build():
    return subprocess.run(["cmake", "--build", str(MUT_CPU_BUILD), "--config", "Release", "--target", "unit_tests",
                           "-j", "8"], capture_output=True, text=True)


def sync_file(src, dst):
    """Copies src to dst when their contents differ, with a fresh timestamp.
    Keeping src's own time (copy2) once hid a change from MSBuild: the copy's
    tree held objects compiled later than src was edited, but from the
    previous copy, so they looked up to date."""
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def snapshot_sources(dest=MUT_SRC):
    """Copies what the build needs into dest: changed files only, each with a
    fresh timestamp, so the copy's tree rebuilds only what changed since the
    last campaign."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["CMakeLists.txt", "CMakePresets.json"]:
        sync_file(ROOT / name, dest / name)
    for d in ["src", "tests"]:
        for src in sorted((ROOT / d).rglob("*")):
            if src.is_file():
                sync_file(src, dest / src.relative_to(ROOT))


def gpu_configure(keep_snapshot=False):
    if not keep_snapshot or not (MUT_SRC / "CMakeLists.txt").exists():
        snapshot_sources()
    subprocess.run(["cmake", "-S", str(MUT_SRC), "-B", str(MUT_BUILD), "-DCHESSGPU_BOARDS=quad",
                    "-DCHESSGPU_MODELS=thread;warp", "-DCHESSGPU_EVALS=pesto", "-DCHESSGPU_ORDERINGS=staged",
                    "-DCHESSGPU_STACKS=local",
                    "-DCHESSGPU_FEATURES=none;tt;tt-killers;tt-ext-killers;tt-killers-see;tt-ext-killers-see",
                    "-DCHESSGPU_TT_LAYOUTS=wide128", "-DCHESSGPU_ATTACKS=hq"], check=True, capture_output=True)


def gpu_build():
    return subprocess.run(["cmake", "--build", str(MUT_BUILD), "--config", "Release", "--target",
                           "gpu_tests", "-j", "16"], capture_output=True, text=True)


def run_tests(exe, filters, timeout):
    """Returns the verdict for one planted bug."""
    failed = []
    for flt in filters:
        try:
            r = subprocess.run([str(exe), flt], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            failed.append(f"{flt}: hung")
            continue
        if r.returncode != 0:
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            fails = [l.split("]")[1].split("(")[0].strip() for l in r.stdout.splitlines() if l.startswith("[ FAIL ]")]
            failed.append(f"{flt}: {tail} {fails}")
    return ("CAUGHT " + "; ".join(failed)) if failed else "MISSED"


def baseline(exe, filters, timeout):
    """The unmodified build must pass its tests, or every planted bug would
    look caught."""
    verdict = run_tests(exe, filters, timeout)
    if verdict != "MISSED":
        raise SystemExit(f"the unmodified sources fail their tests ({exe}): {verdict}")


def campaign(name, mutations, build, exe, filters_of, timeout, source_root=ROOT):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    log_path.write_text("", encoding="utf-8")

    def log(text):
        print(text, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    caught = 0
    for m in mutations:
        title, path, old, new = m[:4]
        p = source_root / path
        original = p.read_text(encoding="utf-8")
        try:
            if original.count(old) != 1:
                log(f"{title}: pattern found {original.count(old)} times, skipped")
                continue
            p.write_text(original.replace(old, new), encoding="utf-8")
            b = build()
            if b.returncode != 0:
                errors = [l for l in b.stdout.splitlines() if "error" in l][:2]
                verdict = "build failed: " + " | ".join(errors)
            else:
                verdict = run_tests(exe, filters_of(m), timeout)
        finally:
            p.write_text(original, encoding="utf-8")
        caught += verdict.startswith("CAUGHT")
        log(f"{title}: {verdict}")
    build()  # the unmodified sources again
    log(f"== {caught} of {len(mutations)} caught")


MP = "src/host/multipv.cpp"
MULTIPV_CPU = [
    ("a child's mate score keeps its distance", MP,
     "    if (s > kScoreMateBound) return -s + 1;\n    if (s < -kScoreMateBound) return -s - 1;\n    return -s;\n}\n\nint score_to_child",
     "    return -s;\n}\n\nint score_to_child"),
    ("the bound for a child keeps the mate distance", MP,
     "    if (s > kScoreMateBound) return -s - 1;\n    if (s < -kScoreMateBound) return -s + 1;\n    return -s;\n}\n\nSearchOutcome",
     "    return -s;\n}\n\nSearchOutcome"),
    ("a move beating the worst line keeps its zero-window score", MP,
     "                if (backend.honors_window()) {", "                if (false) {"),
    ("the zero window is around the best line, not the worst", MP,
     "score_to_child(lines.back().score)", "score_to_child(lines.front().score)"),
    ("the root isn't added to a child's history", MP,
     "        c.history.entries.push_back(parent);\n", ""),
    ("earlier game positions keep their plies", MP,
     "for (HistoryEntry& e : c.history.entries) e.ply = short(e.ply - 1);",
     "for (HistoryEntry& e : c.history.entries) e.ply = short(e.ply);"),
    ("returning to a child's position isn't a draw", MP,
     "        c.history.root_repeated = true;", "        c.history.root_repeated = false;"),
    ("a second occurrence counts as a draw", MP,
     "        if (h.entries[size_t(e)].key == key) return h.entries[size_t(e)].repeated != 0;",
     "        if (h.entries[size_t(e)].key == key) return true;"),
    ("repetitions compared one ply off", MP,
     "    int distance = 2;  // of entry e from the position", "    int distance = 1;  // of entry e from the position"),
    ("insufficient material isn't a draw", MP,
     " || insufficient_material(child) ||", " ||"),
    ("a mate on the hundredth half-move counts as a draw", MP,
     "(halfmove >= 100 && count_legal(child, a) > 0)", "(halfmove >= 100 && count_legal(child, a) >= 0)"),
    ("the move number stays after Black's move", MP,
     "root.fullmove + (state::side(b.state()) == BLACK ? 1 : 0)", "root.fullmove"),
    ("a stop during depth 1 leaves no move", MP,
     "        if (!finished) {\n            if (depth > 1) break;", "        if (!finished) {\n            break;"),
    ("more lines than asked for", MP,
     "        if (lines.size() > wanted) lines.pop_back();\n", ""),
    ("lines ordered worst first", MP,
     "[&](const RootLine& l) { return line.score > l.score; }", "[&](const RootLine& l) { return line.score < l.score; }"),
    ("a draw on arrival is searched", MP, "        if (c->draw) {", "        if (false) {"),
    ("MultiPV 1 takes the MultiPV path", SB,
     "    if (request.multipv > 1) return multipv_deepening", "    if (request.multipv >= 1) return multipv_deepening"),
]

GUI_ANALYSIS_TESTS = ["tests/test_tree.py", "tests/test_pgn.py", "tests/test_game.py", "tests/test_review.py",
                      "tests/test_widgets.py", "tests/test_main_window.py"]
GUI_ANALYSIS = [
    # PGN trees.
    ("pgn: the last variation written as the main move", G + "pgn.py",
     "        main = node.children[0]\n        tokens += _move_tokens(main, pos, number)",
     "        main = node.children[-1]\n        tokens += _move_tokens(main, pos, number)"),
    ("pgn: variations without parentheses", G + "pgn.py",
     "            tokens[first] = \"(\" + tokens[first]\n", ""),
    ("pgn: NAGs not written", G + "pgn.py", "    tokens += [f\"${n}\" for n in node.nags]\n", ""),
    ("pgn: comments before a variation not written", G + "pgn.py",
     "            tokens += _comment_tokens(variation.starting_comment)\n", ""),
    ("pgn: the game's comment not written", G + "pgn.py",
     "    tokens = _comment_tokens(game.root.comment)\n", "    tokens = []\n"),
    ("pgn: Black's move after a variation without its number", G + "pgn.py",
     "            tokens[-1] += \")\"\n            number = True\n", "            tokens[-1] += \")\"\n"),
    ("reader: a variation continues the move instead of replacing it", G + "pgn.py",
     "        self.line, self.last, self.pending = self.parent[id(self.last)], None, \"\"",
     "        self.line, self.last, self.pending = self.last, None, \"\""),
    ("reader: annotation glyphs on moves ignored", G + "pgn.py",
     "        if glyph in GLYPH_NAGS:\n            node.nags.append", "        if False:\n            node.nags.append"),
    ("reader: a comment before a variation's first move lost", G + "pgn.py",
     "            node.starting_comment, self.pending = self.pending, \"\"", "            self.pending = \"\""),
    ("reader: an illegal move in a variation fails the game", G + "pgn.py",
     "            if not self.stack:\n                raise PgnError", "            if True:\n                raise PgnError"),
    ("reader: a dropped variation's inner parentheses miscounted", G + "pgn.py",
     "        if self.skip > 1:\n", "        if self.skip > 2:\n"),
    ("reader: repeated commands overwrite each other", G + "pgn.py",
     "            commands[name] = f\"{commands[name]},{value}\" if commands.get(name) else value",
     "            commands[name] = value"),
    # The record's tree.
    ("tree: a move already there is added again", G + "game.py",
     "        node = parent.child(move)\n        if node is None:", "        node = None\n        if node is None:"),
    ("tree: a move at the line's end isn't on the main line", G + "game.py",
     "            if node is parent.children[0]:\n                self._sync()", "            if False:\n                self._sync()"),
    ("tree: promote moves a variation down", G + "game.py",
     "        siblings[i - 1], siblings[i] = siblings[i], siblings[i - 1]", "        siblings.append(siblings.pop(i))"),
    ("tree: make_main stops at the first branch", G + "game.py",
     "            siblings.insert(0, siblings.pop(i))\n            node = node.parent",
     "            siblings.insert(0, siblings.pop(i))\n            break"),
    ("tree: delete keeps the node", G + "game.py",
     "        parent.children = [c for c in parent.children if c is not node]",
     "        parent.children = list(parent.children)"),
    ("tree: taking back leaves the variations", G + "game.py",
     "        (self.moves[ply - 1] if ply > 0 else self.root).children.clear()",
     "        del (self.moves[ply - 1] if ply > 0 else self.root).children[:1]"),
    ("tree: detached nodes still count as in the game", G + "game.py",
     "            if not any(c is node for c in node.parent.children):\n                return False\n", ""),
    ("tree: repetitions miss the oldest position", G + "game.py",
     "        for _ in range(pos.halfmove + 1):", "        for _ in range(pos.halfmove):"),
    ("record: clocks not saved", G + "game.py",
     "            commands[\"clk\"] = format_clock_command(node.clock_ms)", "            pass"),
    ("record: arrows and circles not saved", G + "game.py",
     "        commands.update(shapes_commands(node.arrows, node.circles))\n", ""),
    ("record: review evaluations not saved", G + "game.py",
     "            commands[\"eval\"] = node.review.command()", "            pass"),
    ("record: match runners' notes taken as comments", G + "game.py",
     "        node.comment = \", \".join(p for p in parts if not _NOTE_RE.match(p))",
     "        node.comment = \", \".join(parts)"),
    ("record: a reason with a comma cut short", G + "game.py",
     "                    reason = \", \".join(parts[i:])", "                    reason = parts[-1]"),
    ("record: any last comment taken as the reason", G + "game.py",
     "                if _REASON_RE.match(p):", "                if True:"),
    ("record: variations not saved", G + "game.py",
     "            for child in node.children:\n                c = out.add", "            for child in node.children[:1]:\n                c = out.add"),
    ("record: variations not read", G + "game.py",
     "            for child in src.children:", "            for child in src.children[:1]:"),
    ("record: clock tenths dropped", G + "game.py", " + (f\".{t}\" if t else \"\")", ""),
    ("record: arrows read as circles", G + "game.py",
     "        if m.group(\"b\") and m.group(\"b\") != m.group(\"a\"):", "        if False:"),
    ("explore: allowed during a game", G + "game.py",
     "        if self.status in (self.RUNNING, self.PREPARING) or not self.record.contains(node):",
     "        if not self.record.contains(node):"),
    ("edit: allowed during a game", G + "game.py",
     "        if self.status in (self.RUNNING, self.PREPARING):\n            return None\n        result = change(*args)",
     "        result = change(*args)"),
    ("score: [%eval] mates misread", G + "score.py", "    if m.group(\"mate\") is not None:", "    if False:"),
    ("score: [%eval] mates lose their sign", G + "score.py",
     "            text = f\"#{'' if s.winning() else '-'}{abs(s.mate)}\"", "            text = f\"#{abs(s.mate)}\""),
    ("score: a move's time alone not read", G + "score.py",
     "        return MoveEval(None, None, float(t.group(\"seconds\"))) if t else None", "        return None"),
    ("rules: a pass keeps the side to move", G + "rules.py",
     "        return Position(self.board, other(self.turn), self.castling, None,",
     "        return Position(self.board, self.turn, self.castling, None,"),
    # The move tree view.
    ("tree view: variations not shown", G + "move_tree.py",
     "            variations = node.parent.children[1:]\n", "            variations = []\n"),
    ("tree view: comments not shown", G + "move_tree.py",
     "                if node.comment:\n                    rows.append(self._wide_row(self._comment(node.comment)))",
     "                if False:\n                    rows.append(self._wide_row(self._comment(node.comment)))"),
    ("tree view: the shown move not highlighted", G + "move_tree.py",
     "        if node is self.current:\n            style +=", "        if False:\n            style +="),
    ("tree view: variations inside variations not shown", G + "move_tree.py",
     "            if node is not first:\n                for v in", "            if False:\n                for v in"),
    ("tree view: move glyphs not shown", G + "move_tree.py",
     "    move = \"\".join(MOVE_NAGS[n] for n in nags if n in MOVE_NAGS)", "    move = \"\""),
    ("tree view: two move glyphs at once", G + "move_tree.py",
     "    return [n for n in nags if n not in group] + [nag]", "    return nags + [nag]"),
    ("tree view: a line starting with Black has no number", G + "move_tree.py",
     "        elif i == 0:\n            parts.append(f\"{before.fullmove}...\")",
     "        elif False:\n            parts.append(f\"{before.fullmove}...\")"),
    ("tree view: a click selects the move before", G + "move_tree.py",
     "            self.node_selected.emit(self._nodes[i])", "            self.node_selected.emit(self._nodes[i - 1])"),
    ("tree view: main-line moves can be promoted", G + "move_tree.py",
     "(\"promote\", \"Promote variation\", not main and self.can_edit)", "(\"promote\", \"Promote variation\", self.can_edit)"),
    ("comment box: edits not stored", G + "move_tree.py",
     "            self.node.comment = self.toPlainText().strip()", "            pass"),
    # Engine output, graph.
    ("engine output: lines beyond the count kept", G + "panels.py",
     "        if info.pv and rank <= self.max_lines:", "        if info.pv:"),
    ("engine output: a bound replaces a line's score", G + "panels.py",
     "white_score(info.score, root.turn) if exact else self.lines.get(rank, (None, []))[0]",
     "white_score(info.score, root.turn) if info.score is not None else None"),
    ("engine output: another position's lines kept", G + "panels.py",
     "            self.root, self.lines = root, {}", "            self.root = root"),
    ("engine output: a click reports the wrong line", G + "panels.py",
     "            self.pv_clicked.emit(int(rank), int(count))", "            self.pv_clicked.emit(int(count), int(rank))"),
    ("node score: an engine's score not seen from White", G + "panels.py",
     "        return white_score(node.eval.score, node.parent.position.turn)", "        return node.eval.score"),
    # Review.
    ("review: the engine's own move judged", G + "review.py",
     "        if best == node.move.uci():", "        if False:"),
    ("review: losses seen from White's side", G + "review.py",
     "            loss = max(0.0, expected_score(before.score, mover) - expected_score(after.score, mover))",
     "            loss = max(0.0, expected_score(before.score, WHITE) - expected_score(after.score, WHITE))"),
    ("review: mistakes count as blunders", G + "review.py", "    if loss >= BLUNDER:", "    if loss >= MISTAKE:"),
    ("review: accuracy averaged over one move too few", G + "review.py",
     "            s.accuracy /= s.moves", "            s.accuracy /= max(1, s.moves - 1)"),
    ("annotate: other move NAGs kept", G + "review.py",
     "        node.nags = [n for n in node.nags if n > 7] + [KIND_NAGS[j.kind]]",
     "        node.nags = node.nags + [KIND_NAGS[j.kind]]"),
    ("annotate: the comment added again", G + "review.py", "        if text not in node.comment:", "        if True:"),
    ("review controller: Black's scores not turned", G + "review.py",
     "            score = info.score if turn == WHITE else info.score.negated()", "            score = info.score"),
    ("review controller: a mate scored for the wrong side", G + "review.py",
     "                score = Score(mate=0) if pos.turn == WHITE else Score(mate=0).negated()",
     "                score = Score(mate=0)"),
    ("review dialog: depths become movetimes", G + "review_panel.py",
     "        return self.engine.currentText(), movetime, depth, self.annotate.isChecked()",
     "        return self.engine.currentText(), movetime or 1000, None, self.annotate.isChecked()"),
    ("review panel: good moves listed", G + "review_panel.py",
     "            if j.kind is None:\n                continue\n            item", "            item"),
    # Analysis controller and the window.
    ("analysis: MultiPV not set on the engine", G + "game.py",
     "        if self.engine is None or not self.engine.set_option(\"MultiPV\", self.lines):",
     "        if True:"),
    ("analysis: no new search after a line count change", G + "game.py",
     "        if self.running:  # analyse again with the new count", "        if False:  # analyse again with the new count"),
    ("analysis: the line count missing at start-up", G + "game.py",
     "        if self.lines > 1:\n            options[\"MultiPV\"]", "        if False:\n            options[\"MultiPV\"]"),
    ("window: board moves explore from the game's end", G + "main_window.py",
     "        node = g.explore(self.view_node, move)", "        node = g.explore(g.record.end, move)"),
    ("window: navigation ignores the shown variation", G + "main_window.py",
     "        line = self.view_node.line()\n        self.go_to_node(line[", "        line = self.game.record.root.line()\n        self.go_to_node(line["),
    ("window: switching variations does nothing", G + "main_window.py",
     "        if 0 <= i + delta < len(siblings):", "        if False:"),
    ("window: drawn shapes not stored", G + "main_window.py",
     "        self.view_node.set_shapes(self.board.shape_arrows, self.board.shape_circles)", "        pass"),
    ("window: deleting the shown line jumps to the game's end", G + "main_window.py",
     "                self.view_node, self._follow = node.parent, False", "                pass"),
    ("window: threat mode analyses the position itself", G + "main_window.py",
     "                after_pass = node.position.null_move()", "                after_pass = node.position"),
    ("window: a line click plays one move too many", G + "main_window.py",
     "        for text in out.lines[rank][1][:count]:", "        for text in out.lines[rank][1][:count + 1]:"),
    ("window: a stale hint shown", G + "main_window.py",
     "        if best is None or node is None or node is not g.record.end or not g.human_to_move:",
     "        if best is None or node is None:"),
    ("window: the line count not saved", G + "main_window.py",
     "        self.settings.setValue(\"analysis_lines\", n)", "        pass"),
    ("window: a review outlives its game", G + "main_window.py",
     "        if self.review.running and record is not self._review_record:", "        if False:"),
    ("window: a review doesn't annotate", G + "main_window.py",
     "            if complete and self._review_annotate:", "            if False:"),
]


GUI_EVENTS_TESTS = ["tests/test_tournament.py", "tests/test_database.py", "tests/test_game.py",
                    "tests/test_widgets.py", "tests/test_main_window.py"]
GUI_EVENTS = [
    # Tournaments.
    ("schedule: colors not swapped", G + "tournament.py",
     "            out.append(Pairing(len(out) + 1, b, a, o))", "            out.append(Pairing(len(out) + 1, a, b, o))"),
    ("schedule: a gauntlet pairs everyone", G + "tournament.py",
     "        pairs = [(engines[0], e) for e in engines[1:]]",
     "        pairs = [(a, b) for i, a in enumerate(engines) for b in engines[i + 1:]]"),
    ("schedule: one opening for every pairing", G + "tournament.py", "            opening += 1", "            opening += 0"),
    ("openings: a PGN's games taken whole", G + "tournament.py",
     "            for m in g.moves[:max(0, plies)]:", "            for m in g.moves:"),
    ("openings: an EPD's move counters dropped", G + "tournament.py",
     "            fen = \" \".join(fields[:6]) if len(fields) >= 6 and fields[4].isdigit() else \" \".join(fields[:4]) + \" 0 1\"",
     "            fen = \" \".join(fields[:4]) + \" 0 1\""),
    ("openings: the order ignores the seed", G + "tournament.py",
     "        random.Random(seed).shuffle(out)", "        random.Random().shuffle(out)"),
    ("elo: the sign flipped", G + "tournament.py",
     "    return -400.0 * math.log10(1.0 / score - 1.0)", "    return 400.0 * math.log10(1.0 / score - 1.0)"),
    ("elo: a binomial error bar", G + "tournament.py",
     "    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s ** 2) / n", "    var = s * (1 - s)"),
    ("los: upside down", G + "tournament.py",
     "    return 0.5 + 0.5 * math.erf(", "    return 0.5 - 0.5 * math.erf("),
    ("standings: draws count as losses", G + "tournament.py",
     "            if r.result == \"1/2-1/2\":\n                s.draws += 1", "            if False:\n                s.draws += 1"),
    ("standings: unfinished games counted", G + "tournament.py",
     "    return [r for r in results if r.result in (\"1-0\", \"0-1\", \"1/2-1/2\")]", "    return list(results)"),
    ("cross-table: Black's points from White's side", G + "tournament.py",
     "(r.black, r.white, 1.0 - white_points)", "(r.black, r.white, white_points)"),
    ("adjudication: one report enough to win", G + "tournament.py",
     "            if all(v >= r.resign_cp for v in last):", "            if any(v >= r.resign_cp for v in last):"),
    ("adjudication: draws before the threshold move", G + "tournament.py",
     " and record.position.fullmove >= r.draw_from_move:", ":"),
    ("adjudication: the move limit a move early", G + "tournament.py",
     "        if r.max_moves and record.position.fullmove > r.max_moves:",
     "        if r.max_moves and record.position.fullmove >= r.max_moves:"),
    ("controller: engines not kept between games", G + "tournament.py",
     "        self.game.keep_engines(True)\n        self.save_state()", "        self.save_state()"),
    ("controller: a pause doesn't wait", G + "tournament.py",
     "        if self._pause_requested:\n            self.state = self.PAUSED\n            self.changed.emit()\n        elif result",
     "        if False:\n            self.state = self.PAUSED\n            self.changed.emit()\n        elif result"),
    ("controller: finished games not written", G + "tournament.py",
     "            self._append_pgn(record)\n", ""),
    ("controller: the state not saved", G + "tournament.py",
     "            path.write_text(json.dumps(data, indent=1), encoding=\"utf-8\")", "            pass"),
    ("controller: resuming replays finished games", G + "tournament.py",
     "        return [p for p in self.pairings if p.number not in done]", "        return list(self.pairings)"),
    ("controller: no adjudication", G + "tournament.py",
     "        if verdict is not None:\n            QTimer.singleShot", "        if False:\n            QTimer.singleShot"),
    ("controller: no Round tag", G + "tournament.py",
     "        self.game.record.tags[\"Round\"] = str(p.number)\n", ""),
    # The game controller: kept engines, adjudication, the book.
    ("engines: the kept ones never used", G + "game.py",
     "            if e is None and self.engine_cache is not None:\n                kept",
     "            if False:\n                kept"),
    ("adjudicate: also when no game runs", G + "game.py",
     "        if self.status != self.RUNNING:\n            return False\n        self._finish(result, reason)",
     "        self._finish(result, reason)"),
    ("book: used past its moves", G + "game.py",
     "        if self.book is None or len(self.record.moves) >= self.book_plies:", "        if self.book is None:"),
    ("book: its moves not marked", G + "game.py",
     "        self._play(move, None, comment=\"book\")", "        self._play(move, None)"),
    # Databases.
    ("scan: one tag per line", G + "database.py",
     "            for m in _TAG_RE.finditer(stripped):", "            for m in list(_TAG_RE.finditer(stripped))[:1]:"),
    ("scan: escapes kept", G + "database.py",
     "                tags[m.group(1)] = re.sub(r\"\\\\(.)\", r\"\\1\", m.group(2))", "                tags[m.group(1)] = m.group(2)"),
    ("scan: games run together", G + "database.py",
     "            if moves_seen:\n                flush()", "            if False:\n                flush()"),
    ("main line: variations read into it", G + "database.py",
     "        elif kind == \"move\" and depth == 0:", "        elif kind == \"move\":"),
    ("main line: one move past the limit", G + "database.py",
     "            if len(moves) >= limit:", "            if len(moves) > limit:"),
    ("index: results in the wrong column", G + "database.py",
     "RESULT_SLOTS = {\"1-0\": 0, \"1/2-1/2\": 1, \"0-1\": 2}", "RESULT_SLOTS = {\"1-0\": 2, \"1/2-1/2\": 1, \"0-1\": 0}"),
    ("index: a game's last position left out", G + "database.py",
     "        key = explorer_key(pos)\n        if key not in seen:\n            index.games.setdefault(key, []).append(number)\n        index.indexed += 1",
     "        index.indexed += 1"),
    ("index: merging replaces game lists", G + "database.py",
     "            self.games.setdefault(key, []).extend(numbers)", "            self.games[key] = list(numbers)"),
    ("book: every move as likely", G + "database.py",
     "        choices = [(m, sum(s)) for m, s in self.stats(pos) if sum(s) >= min_games]",
     "        choices = [(m, 1) for m, s in self.stats(pos) if sum(s) >= min_games]"),
    ("search: the result ignored", G + "database.py",
     "            if result and g.result != result:", "            if False:"),
    ("search: the position ignored", G + "database.py",
     "            if allowed is not None and g.number not in allowed:", "            if False:"),
    ("games model: always ascending", G + "database_panel.py",
     "                        reverse=order == Qt.SortOrder.DescendingOrder)", "                        reverse=False)"),
    ("explorer: a click plays the first move", G + "database_panel.py",
     "            self.move_selected.emit(self.moves[row])", "            self.move_selected.emit(self.moves[0])"),
    ("explorer: no result shares", G + "database_panel.py",
     "            bar.setData(Qt.ItemDataRole.UserRole, (s[0], s[1], s[2]))", "            bar.setData(Qt.ItemDataRole.UserRole, None)"),
    ("database panel: the position filter lost", G + "database_panel.py",
     "        return self.text.text(), \"\" if result == \"Any result\" else result, self.here.isChecked()",
     "        return self.text.text(), \"\" if result == \"Any result\" else result, False"),
    # Dialogs and panels.
    ("tournament dialog: the gauntlet's engine not first", G + "tournament_panel.py",
     "            names = [first] + [n for n in names if n != first]", "            names = names"),
    ("tournament dialog: adjudication always on", G + "tournament_panel.py",
     "                                      self.adjudicate.isChecked()))", "                                      True))"),
    ("tournament panel: the cross-table transposed", G + "tournament_panel.py",
     "                    points, games = cross.get((a, b), (0.0, 0))", "                    points, games = cross.get((b, a), (0.0, 0))"),
    ("tournament panel: wins shown as points", G + "tournament_panel.py",
     "str(s.losses), f\"{s.points:g}\",", "str(s.losses), f\"{s.wins:g}\","),
    ("new game: the book never chosen", G + "dialogs.py",
     "                             2 * self.book_moves.value() if self.book.isChecked() else 0)",
     "                             0)"),
    ("time control: no increment", G + "dialogs.py",
     "inc_ms=round(self.increment.value() * 1000))", "inc_ms=0)"),
    # The window.
    ("window: a tournament game doesn't block others", G + "main_window.py",
     "        if self.tournament.state == self.tournament.RUNNING:\n            self.show_message(\"A tournament",
     "        if False:\n            self.show_message(\"A tournament"),
    ("window: the wrong tournament game shown", G + "main_window.py",
     "            if g.tags.get(\"Round\") == str(number) and", "            if g.tags.get(\"Round\") == str(number + 1) and"),
    ("window: explorer moves ignored", G + "main_window.py",
     "        if self.board.interactive and self.viewed.is_legal(move):", "        if False:"),
    ("window: the database's next game opened", G + "main_window.py",
     "        game = self.database.game(number)\n", "        game = self.database.game(number + 1)\n"),
    ("window: the book not handed to the game", G + "main_window.py",
     "        self.game.book = self.database.index if book_plies else None", "        self.game.book = None"),
]


def gui_snapshot():
    """Copies gui/, the perft suites its rules tests read and the match
    openings (the tournaments' built-in set) into MUT_GUI (changed files
    only, with fresh timestamps)."""
    for folder in (ROOT / "gui", ROOT / "tests" / "perft"):
        for src in sorted(folder.rglob("*")):
            if src.is_file() and not {"__pycache__", ".pytest_cache"} & set(src.parts):
                sync_file(src, MUT_GUI / src.relative_to(ROOT))
    openings = ROOT / "tests" / "search" / "openings.epd"
    sync_file(openings, MUT_GUI / openings.relative_to(ROOT))


def gui_tests(files, timeout):
    """Runs pytest on `files` in the copy; the verdict as for run_tests."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    engine = ROOT / "build" / "Release" / "chessgpu.exe"
    if engine.exists():
        env["CHESSGPU_ENGINE"] = str(engine)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    try:
        r = subprocess.run([str(python), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *files],
                           cwd=MUT_GUI / "gui", capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "CAUGHT pytest: hung"
    if r.returncode == 0:
        return "MISSED"
    failed = [l.split(" - ")[0] for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]
    return "CAUGHT " + ("; ".join(failed[:2]) or r.stdout.strip().splitlines()[-1])


def gui_campaign(name, mutations, files, timeout):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    log_path.write_text("", encoding="utf-8")

    def log(text):
        print(text, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    gui_snapshot()
    verdict = gui_tests(files, timeout)  # the unmodified copy first
    if verdict != "MISSED":
        raise SystemExit(f"the unmodified GUI fails its tests: {verdict}")
    caught = 0
    for title, path, old, new in mutations:
        p = MUT_GUI / path
        original = p.read_text(encoding="utf-8")
        try:
            if original.count(old) != 1:
                log(f"{title}: pattern found {original.count(old)} times, skipped")
                continue
            p.write_text(original.replace(old, new), encoding="utf-8")
            verdict = gui_tests(files, timeout)
        finally:
            p.write_text(original, encoding="utf-8")
        caught += verdict.startswith("CAUGHT")
        log(f"{title}: {verdict}")
    log(f"== {caught} of {len(mutations)} caught")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign",
                    choices=["killers-cpu", "see-cpu", "abdada-cpu", "move32-cpu", "nnue-cpu", "pvs-cpu",
                             "qchecks-cpu", "aspiration-cpu", "multipv-cpu", "gpu-explicit", "gpu-abdada",
                             "gui-look", "gui-analysis", "gui-events", "snapshot"])
    ap.add_argument("--keep-snapshot", action="store_true",
                    help="GPU campaigns: plant bugs in build/mut-src as it is (taken earlier with 'snapshot')")
    ap.add_argument("--only", metavar="TEXT", action="append",
                    help="plant only the bugs whose title contains TEXT (to rerun missed ones; repeatable)")
    a = ap.parse_args()

    def chosen(mutations):
        picked = [m for m in mutations if a.only is None or any(text in m[0] for text in a.only)]
        if not picked:
            raise SystemExit(f"no planted bug's title contains any of {a.only!r}")
        return picked

    name = a.campaign + ("-only" if a.only else "")  # a partial run keeps the full run's log
    if a.campaign == "snapshot":  # just copy the sources, for a later --keep-snapshot run
        snapshot_sources()
        return
    if a.campaign == "gui-look":
        gui_campaign(name, chosen(GUI_LOOK), GUI_LOOK_TESTS, 600)
        return
    if a.campaign == "gui-analysis":
        gui_campaign(name, chosen(GUI_ANALYSIS), GUI_ANALYSIS_TESTS, 900)
        return
    if a.campaign == "gui-events":
        gui_campaign(name, chosen(GUI_EVENTS), GUI_EVENTS_TESTS, 900)
        return
    cpu_campaigns = {
        "killers-cpu": (KILLERS_CPU, ["killers", "picker"], 600),
        "see-cpu": (SEE_CPU, ["see"], 900),
        "abdada-cpu": (ABDADA_CPU, ["abdada"], 300),
        "move32-cpu": (MOVE32_CPU, ["move32"], 600),
        "nnue-cpu": (NNUE_CPU, ["nnue"], 600),
        "pvs-cpu": (PVS_CPU, ["pvs"], 300),
        "qchecks-cpu": (QCHECKS_CPU, ["qchecks"], 600),
        "aspiration-cpu": (ASPIRATION_CPU, ["aspiration", "plan_time"], 300),
        "multipv-cpu": (MULTIPV_CPU, ["multipv"], 600),
    }
    if a.campaign in cpu_campaigns:
        mutations, filters, timeout = cpu_campaigns[a.campaign]
        mutations = chosen(mutations)
        cpu_configure()
        b = cpu_build()  # the unmodified copy first: a planted bug must be the only change
        if b.returncode != 0:
            raise SystemExit(f"the unmodified sources don't build in {MUT_CPU_BUILD}:\n{b.stdout[-3000:]}")
        unit = MUT_CPU_BUILD / "tests" / "Release" / "unit_tests.exe"
        baseline(unit, filters, timeout)
        campaign(name, mutations, cpu_build, unit, lambda m: filters, timeout, source_root=MUT_CPU_SRC)
    else:
        gpu_configure(a.keep_snapshot)
        b = gpu_build()  # the unmodified copy first: a planted bug must be the only change
        if b.returncode != 0:
            raise SystemExit(f"the unmodified sources don't build in {MUT_BUILD}:\n{b.stdout[-3000:]}")
        gpu_exe = MUT_BUILD / "tests" / "Release" / "gpu_tests.exe"
        gpu_mutations = chosen(GPU_EXPLICIT if a.campaign == "gpu-explicit" else GPU_ABDADA)
        baseline(gpu_exe, sorted({m[4] for m in gpu_mutations}) if a.campaign == "gpu-explicit" else ["abdada"], 1800)
        if a.campaign == "gpu-explicit":
            campaign(name, gpu_mutations, gpu_build, gpu_exe, lambda m: [m[4]], 1800, source_root=MUT_SRC)
        else:
            campaign(name, gpu_mutations, gpu_build, gpu_exe, lambda m: ["abdada"], 1800, source_root=MUT_SRC)


if __name__ == "__main__":
    main()
