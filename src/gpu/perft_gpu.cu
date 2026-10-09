#include "gpu/perft_gpu.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "gpu/perft_kernels.cuh"
#include "gpu/variants.h"
#include "host/host_tables.h"
#include "host/profiling.h"
#include "host/uci_move.h"

namespace cb {

bool is_compiled(BoardKind k) {
    return k == BoardKind::Std ? CHESSGPU_WITH_STD_BOARD != 0 : CHESSGPU_WITH_QUAD_BOARD != 0;
}

bool is_compiled(ExecModel m) {
    return m == ExecModel::Thread ? CHESSGPU_WITH_THREAD_MODEL != 0 : CHESSGPU_WITH_WARP_MODEL != 0;
}

bool is_compiled(MovesKind m) {
    return m == MovesKind::Bits16 ? CHESSGPU_WITH_16_MOVES != 0 : CHESSGPU_WITH_32_MOVES != 0;
}

std::vector<MovesKind> compiled_moves() {
    std::vector<MovesKind> v;
    for (const MovesKind m : {MovesKind::Bits16, MovesKind::Bits32})
        if (is_compiled(m)) v.push_back(m);
    return v;
}

bool is_compiled(AttacksKind a) {
    switch (a) {
        case AttacksKind::Hq: return CHESSGPU_WITH_HQ_ATTACKS != 0;
        case AttacksKind::Magic: return CHESSGPU_WITH_MAGIC_ATTACKS != 0;
        case AttacksKind::Kogge: return CHESSGPU_WITH_KOGGE_ATTACKS != 0;
    }
    return false;
}

std::vector<AttacksKind> compiled_attacks() {
    std::vector<AttacksKind> v;
    for (const AttacksKind a : {AttacksKind::Hq, AttacksKind::Magic, AttacksKind::Kogge})
        if (is_compiled(a)) v.push_back(a);
    return v;
}

std::vector<BoardKind> compiled_boards() {
    std::vector<BoardKind> boards;
    for (const BoardKind k : {BoardKind::Std, BoardKind::Quad})
        if (is_compiled(k)) boards.push_back(k);
    return boards;
}

std::vector<ExecModel> compiled_models() {
    std::vector<ExecModel> models;
    for (const ExecModel m : {ExecModel::Thread, ExecModel::Warp})
        if (is_compiled(m)) models.push_back(m);
    return models;
}

namespace {

// The tree expanded breadth-first on the CPU down to `depth` plies.
template <class Board>
struct Split {
    std::vector<Board> tasks;      // positions `depth` plies below the root
    std::vector<int> root_index;   // index in root_moves of each task's first move (-1 if depth == 0)
    std::vector<Move> root_moves;  // legal moves at the root, in generation order
    int depth = 0;
};

// Optional expansions stop before exceeding this many tasks; required ones
// (min_depth) fail beyond kHardMaxTasks rather than exhaust memory.
constexpr size_t kSoftMaxTasks = 8'000'000;
constexpr size_t kHardMaxTasks = 64'000'000;

// Expands at least min_depth and at most max_depth plies, stopping once
// there are target_tasks tasks or the next ply would exceed kSoftMaxTasks.
template <class Board, class Attacks>
Split<Board> split_tree(const Board& root, const Attacks& a, int min_depth, int max_depth, size_t target_tasks) {
    Split<Board> s;
    s.tasks.push_back(root);
    s.root_index.push_back(-1);
    MoveList root_list;
    generate_legal(root, a, root_list);
    s.root_moves.assign(root_list.moves, root_list.moves + root_list.size);

    while (s.depth < max_depth) {
        const bool required = s.depth < min_depth;
        if (!required && s.tasks.size() >= target_tasks) break;
        size_t next_count = 0;
        for (const Board& b : s.tasks) next_count += count_legal(b, a);
        if (!required && next_count > kSoftMaxTasks) break;
        if (next_count > kHardMaxTasks)
            throw std::runtime_error("splitting this perft would create " + std::to_string(next_count) + " tasks");

        std::vector<Board> next;
        std::vector<int> next_index;
        next.reserve(next_count);
        next_index.reserve(next_count);
        for (size_t i = 0; i < s.tasks.size(); ++i) {
            Board b = s.tasks[i];
            MoveList list;
            generate_legal(b, a, list);
            for (int j = 0; j < list.size; ++j) {
                const Undo undo = make_move(b, list.moves[j]);
                next.push_back(b);
                next_index.push_back(s.depth == 0 ? j : s.root_index[i]);
                unmake_move(b, list.moves[j], undo);
            }
        }
        s.tasks = std::move(next);
        s.root_index = std::move(next_index);
        ++s.depth;
    }
    return s;
}

template <class Board>
using PerftKernel = void (*)(const Board*, unsigned, int, gpu::AttackSources, unsigned*, unsigned long long*);

// Only compiled-in models are instantiated; gpu_perft rejects the others first.
template <class Board, class Attacks, class Moves>
PerftKernel<Board> perft_kernel_for(ExecModel model) {
#if CHESSGPU_WITH_THREAD_MODEL
    if (model == ExecModel::Thread) return &gpu::perft_thread_kernel<Board, Attacks, Moves>;
#endif
#if CHESSGPU_WITH_WARP_MODEL
    if (model == ExecModel::Warp) return &gpu::perft_warp_kernel<Board, Attacks, Moves>;
#endif
    throw std::invalid_argument(std::string("execution model '") + name_of(model) + "' is not compiled in");
}

template <class Board, class Attacks, class Moves>
GpuPerftResult run_perft(const DeviceContext& ctx, const Position& root, int depth, const GpuPerftOptions& opt) {
    const Stopwatch total;
    GpuPerftResult result;
    if (depth < 0) throw std::invalid_argument("depth must be >= 0");
    if (opt.block_size < 32 || opt.block_size > 1024 || opt.block_size % 32 != 0)
        throw std::invalid_argument("block size must be a multiple of 32 between 32 and 1024");
    if (depth == 0) {
        result.nodes = 1;
        result.stats.total_ms = total.elapsed_ms();
        return result;
    }

    // Launch configuration: as many blocks as fit on the GPU at once.
    const bool warp_model = opt.model == ExecModel::Warp;
    const PerftKernel<Board> kernel = perft_kernel_for<Board, Attacks, Moves>(opt.model);
    const gpu::AttackSources sources{ctx.device_tables(),
                                     opt.attacks == AttacksKind::Magic ? ctx.device_magic_tables() : nullptr};
    const size_t shared_bytes =
        warp_model ? size_t(opt.block_size / 32) * sizeof(gpu::WarpLists<typename Moves::MoveType>) : 0;
    if (shared_bytes > ctx.info().shared_per_block_optin)
        throw std::invalid_argument("block size needs " + std::to_string(shared_bytes) +
                                    " B of shared memory; the GPU allows " +
                                    std::to_string(ctx.info().shared_per_block_optin));
    if (shared_bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, opt.block_size, shared_bytes));
    if (blocks_per_sm == 0) throw std::runtime_error("the perft kernel does not fit on an SM with these settings");
    const int blocks = blocks_per_sm * ctx.info().sm_count;
    const size_t workers = warp_model ? size_t(blocks) * (opt.block_size / 32) : size_t(blocks) * opt.block_size;

    // Split depth: every task keeps at least one ply for the GPU (unless a
    // depth-1 divide needs the root's children as tasks), and at most
    // kGpuMaxTaskDepth plies.
    int min_split = std::max(0, depth - kGpuMaxTaskDepth);
    if (opt.divide) min_split = std::max(min_split, 1);
    const int max_split = std::max(depth - 1, min_split);
    int split_min = min_split;
    int split_max = max_split;
    size_t target_tasks = 0;
    if (opt.split_depth >= 0) {
        if (opt.split_depth < min_split || opt.split_depth > max_split)
            throw std::invalid_argument("split depth must be between " + std::to_string(min_split) + " and " +
                                        std::to_string(max_split) + " for this depth");
        split_min = split_max = opt.split_depth;
    } else {
        // Enough tasks to keep every worker busy while sizes vary, leaving
        // the GPU at least 2 plies when the depth allows it.
        if (opt.tasks_per_worker < 1) throw std::invalid_argument("tasks per worker must be >= 1");
        target_tasks = size_t(opt.tasks_per_worker) * workers;
        split_max = std::max(min_split, std::min(max_split, depth - 2));
    }

    const HqAttacks host_attacks(host_attack_tables());
    Split<Board> split;
    {
        const NvtxRange range("split");
        const Stopwatch timer;
        split = split_tree(to_board<Board>(root), host_attacks, split_min, split_max, target_tasks);
        result.stats.split_ms = timer.elapsed_ms();
    }
    const size_t num_tasks = split.tasks.size();
    if (num_tasks > 0xFFFFFFFFull / 2) throw std::runtime_error("too many tasks");
    result.stats.split_depth = split.depth;
    result.stats.tasks = num_tasks;
    result.stats.blocks = blocks;
    result.stats.block_size = opt.block_size;
    result.stats.blocks_per_sm = blocks_per_sm;

    std::vector<unsigned long long> counts(num_tasks, 0);
    if (num_tasks > 0) {
        DeviceBuffer<Board> d_tasks(num_tasks);
        DeviceBuffer<unsigned long long> d_counts(num_tasks);
        DeviceBuffer<unsigned> d_next(1);
        {
            const NvtxRange range("upload");
            const Stopwatch timer;
            d_tasks.upload(split.tasks);
            d_next.zero();
            result.stats.upload_ms = timer.elapsed_ms();
        }
        {
            const NvtxRange range("kernel");
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            kernel<<<blocks, opt.block_size, shared_bytes>>>(d_tasks.get(), unsigned(num_tasks), depth - split.depth,
                                                              sources, d_next.get(), d_counts.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            result.stats.kernel_ms = ms;
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        {
            const NvtxRange range("download");
            const Stopwatch timer;
            d_counts.download(counts.data(), num_tasks);
            result.stats.download_ms = timer.elapsed_ms();
        }
    }

    for (const unsigned long long c : counts) result.nodes += c;
    if (opt.divide) {
        std::vector<u64> per_root(split.root_moves.size(), 0);
        for (size_t i = 0; i < num_tasks; ++i) per_root[split.root_index[i]] += counts[i];
        for (size_t i = 0; i < split.root_moves.size(); ++i) result.divide.emplace_back(split.root_moves[i], per_root[i]);
        std::sort(result.divide.begin(), result.divide.end(),
                  [](const auto& x, const auto& y) { return move_to_uci(x.first) < move_to_uci(y.first); });
    }
    result.stats.total_ms = total.elapsed_ms();
    return result;
}

template <class Board, class Moves>
GpuPerftResult run_perft_with_attacks(const DeviceContext& ctx, const Position& root, int depth,
                                      const GpuPerftOptions& opt) {
    switch (opt.attacks) {
#if CHESSGPU_WITH_HQ_ATTACKS
        case AttacksKind::Hq: return run_perft<Board, HqAttacks, Moves>(ctx, root, depth, opt);
#endif
#if CHESSGPU_WITH_MAGIC_ATTACKS
        case AttacksKind::Magic: return run_perft<Board, MagicAttacks, Moves>(ctx, root, depth, opt);
#endif
#if CHESSGPU_WITH_KOGGE_ATTACKS
        case AttacksKind::Kogge: return run_perft<Board, KoggeStoneAttacks, Moves>(ctx, root, depth, opt);
#endif
        default: break;
    }
    throw std::invalid_argument("unknown attacks kind");
}

template <class Board>
GpuPerftResult run_perft_with_moves(const DeviceContext& ctx, const Position& root, int depth,
                                    const GpuPerftOptions& opt) {
#if CHESSGPU_WITH_16_MOVES
    if (opt.moves == MovesKind::Bits16) return run_perft_with_attacks<Board, Moves16>(ctx, root, depth, opt);
#endif
#if CHESSGPU_WITH_32_MOVES
    if (opt.moves == MovesKind::Bits32) return run_perft_with_attacks<Board, Moves32>(ctx, root, depth, opt);
#endif
    throw std::invalid_argument("unknown move encoding");
}

}  // namespace

GpuPerftResult gpu_perft(const DeviceContext& ctx, const Position& root, int depth, const GpuPerftOptions& options) {
    if (!is_compiled(options.board))
        throw std::invalid_argument(std::string("board '") + name_of(options.board) +
                                    "' is not compiled into this build (CMake option CHESSGPU_BOARDS)");
    if (!is_compiled(options.model))
        throw std::invalid_argument(std::string("execution model '") + name_of(options.model) +
                                    "' is not compiled into this build (CMake option CHESSGPU_MODELS)");
    if (!is_compiled(options.attacks))
        throw std::invalid_argument(std::string("attacks '") + name_of(options.attacks) +
                                    "' are not compiled into this build (CMake option CHESSGPU_ATTACKS)");
    if (!is_compiled(options.moves))
        throw std::invalid_argument(std::string(name_of(options.moves)) +
                                    "-bit moves are not compiled into this build (CMake option CHESSGPU_MOVES)");
    switch (options.board) {
#if CHESSGPU_WITH_STD_BOARD
        case BoardKind::Std:
            return run_perft_with_moves<StdBoard>(ctx, root, depth, options);
#endif
#if CHESSGPU_WITH_QUAD_BOARD
        case BoardKind::Quad:
            return run_perft_with_moves<QuadBoard>(ctx, root, depth, options);
#endif
        default:
            break;
    }
    throw std::invalid_argument("unknown board kind");
}

}  // namespace cb
