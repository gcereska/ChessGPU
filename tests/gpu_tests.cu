// GPU tests: device code must match the CPU, and GPU perft must match the
// perft suites in every board/model combination. Search tests are in
// gpu_search_tests.cu.
//
//   gpu_tests.exe [--quick] [filter]
// --quick shrinks the workloads (for compute-sanitizer, which is ~100x slower).

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#include "core/perft.h"
#include "gpu/device_buffer.h"
#include "gpu/device_context.h"
#include "gpu/perft_gpu.h"
#include "gpu/perft_kernels.cuh"
#include "gpu/variants.h"
#include "host/epd.h"
#include "host/host_tables.h"
#include "host/perft_cpu.h"
#include "test_harness.h"
#include "gpu_test_env.h"
#include "test_util.h"

using namespace cb;

namespace {

bool g_quick = false;

const DeviceContext& context() {
    static const DeviceContext ctx;
    return ctx;
}

const HqAttacks& host_attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

// Board for tests that aren't about boards: the default, if it is compiled in.
BoardKind test_board() { return is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front(); }

}  // namespace

namespace gpu_test_env {
bool quick() { return g_quick; }
const DeviceContext& context() { return ::context(); }
}  // namespace gpu_test_env

// Kernels live in a named namespace: nvcc's stub for a kernel with internal
// linkage trips MSVC warning C4211 at /W4.
namespace gpu_tests {

struct AttackQuery {
    Square square;
    u64 occupied;
};

struct AttackAnswer {
    u64 rook, bishop, queen, knight, king, white_pawn, black_pawn;
};

__global__ void attack_kernel(const AttackTables* tables, const AttackQuery* queries, int n, AttackAnswer* out) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const HqAttacks a(*tables);
    const AttackQuery q = queries[i];
    out[i] = {a.rook(q.square, q.occupied), a.bishop(q.square, q.occupied), a.queen(q.square, q.occupied),
              a.knight(q.square),           a.king(q.square),             a.pawn(WHITE, q.square),
              a.pawn(BLACK, q.square)};
}

__global__ void between_line_kernel(const AttackTables* tables, u64* between, u64* line) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 64 * 64) return;
    const HqAttacks a(*tables);
    between[i] = a.between(Square(i / 64), Square(i % 64));
    line[i] = a.line(Square(i / 64), Square(i % 64));
}

// One thread per position: legal move count and perft to `depth`.
template <class Board>
__global__ void thread_count_kernel(const AttackTables* tables, const Board* boards, int n, int depth,
                                    unsigned* counts, unsigned long long* perfts) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const HqAttacks a(*tables);
    counts[i] = count_legal(boards[i], a);
    PerftStack<4> stack;
    perfts[i] = perft_iterative(boards[i], a, depth, stack);
}

// One warp per position: the warp model's cooperative count and generation.
template <class Board>
__global__ void warp_movegen_kernel(const AttackTables* tables, const Board* boards, int n, unsigned* counts,
                                    int* sizes, Move* lists) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    Move* list = reinterpret_cast<Move*>(shared_bytes) + (threadIdx.x / 32) * gpu::kMaxMovesPerPosition;
    const int lane = int(threadIdx.x & 31);
    const int i = int((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    if (i >= n) return;  // whole warps exit together
    const HqAttacks a(*tables);
    const unsigned count = gpu::warp_count_legal(boards[i], a, lane);
    const int size = gpu::warp_generate_legal<Board, HqAttacks, Moves16>(boards[i], a, list, lane);
    if (lane == 0) {
        counts[i] = count;
        sizes[i] = size;
    }
    for (int k = lane; k < size; k += 32) lists[size_t(i) * gpu::kMaxMovesPerPosition + k] = list[k];
}

}  // namespace gpu_tests

namespace {

// Test positions plus every position up to 2 plies below them.
std::vector<Position> many_positions() {
    std::vector<Position> out;
    const HqAttacks& a = host_attacks();
    for (const std::string& fen : testutil::test_fens()) {
        const Position root = testutil::parse_or_die(fen);
        out.push_back(root);
        StdBoard b = to_board<StdBoard>(root);
        MoveList first;
        generate_legal(b, a, first);
        for (int i = 0; i < first.size; ++i) {
            const Undo u1 = make_move(b, first.moves[i]);
            out.push_back(to_position(b));
            MoveList second;
            generate_legal(b, a, second);
            for (int j = 0; j < second.size; j += g_quick ? 7 : 1) {
                const Undo u2 = make_move(b, second.moves[j]);
                out.push_back(to_position(b));
                unmake_move(b, second.moves[j], u2);
            }
            unmake_move(b, first.moves[i], u1);
        }
    }
    return out;
}

template <class Board>
void check_movegen_on_gpu() {
    const std::vector<Position> positions = many_positions();
    std::vector<Board> boards;
    for (const Position& p : positions) boards.push_back(to_board<Board>(p));
    const int n = int(boards.size());
    const int depth = g_quick ? 2 : 3;

    DeviceBuffer<Board> d_boards(boards.size());
    d_boards.upload(boards);
    DeviceBuffer<unsigned> d_counts(n), d_warp_counts(n);
    DeviceBuffer<unsigned long long> d_perfts(n);
    DeviceBuffer<int> d_sizes(n);
    DeviceBuffer<Move> d_lists(size_t(n) * gpu::kMaxMovesPerPosition);
    d_lists.zero();  // kernels fill only each list's used prefix; the whole buffer is copied back

    gpu_tests::thread_count_kernel<<<(n + 127) / 128, 128>>>(context().device_tables(), d_boards.get(), n, depth,
                                                             d_counts.get(), d_perfts.get());
    CUDA_CHECK(cudaGetLastError());
    const int warps_per_block = 4;
    gpu_tests::warp_movegen_kernel<<<(n + warps_per_block - 1) / warps_per_block, 32 * warps_per_block,
                                     warps_per_block * gpu::kMaxMovesPerPosition * sizeof(Move)>>>(
        context().device_tables(), d_boards.get(), n, d_warp_counts.get(), d_sizes.get(), d_lists.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> counts(n), warp_counts(n);
    std::vector<unsigned long long> perfts(n);
    std::vector<int> sizes(n);
    std::vector<Move> lists(size_t(n) * gpu::kMaxMovesPerPosition);
    d_counts.download(counts.data(), n);
    d_warp_counts.download(warp_counts.data(), n);
    d_perfts.download(perfts.data(), n);
    d_sizes.download(sizes.data(), n);
    d_lists.download(lists.data(), lists.size());

    for (int i = 0; i < n; ++i) {
        Board b = boards[i];
        MoveList expected;
        generate_legal(b, host_attacks(), expected);
        const std::string where = std::string(Board::kName) + " " + format_fen(positions[i]);
        CHECK_MSG(counts[i] == unsigned(expected.size), where);
        CHECK_MSG(warp_counts[i] == unsigned(expected.size), where);
        REQUIRE_MSG(sizes[i] == expected.size, where);
        CHECK_MSG(testutil::sorted_raw(&lists[size_t(i) * gpu::kMaxMovesPerPosition], sizes[i]) ==
                      testutil::sorted_raw(expected.moves, expected.size),
                  where);
        CHECK_MSG(perfts[i] == perft_recursive(b, host_attacks(), depth), where);
    }
    CHECK(n > 500);
}

}  // namespace

TEST_CASE(gpu_attacks_match_host) {
    std::vector<gpu_tests::AttackQuery> queries;
    std::mt19937_64 rng(0x7E8);
    for (int i = 0; i < (g_quick ? 2000 : 100000); ++i) {
        u64 occupied = rng() & rng();
        if (i % 3 == 0) occupied &= rng();
        queries.push_back({Square(i % 64), occupied});
    }
    const int n = int(queries.size());
    DeviceBuffer<gpu_tests::AttackQuery> d_queries(queries.size());
    d_queries.upload(queries);
    DeviceBuffer<gpu_tests::AttackAnswer> d_answers(queries.size());
    gpu_tests::attack_kernel<<<(n + 255) / 256, 256>>>(context().device_tables(), d_queries.get(), n, d_answers.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<gpu_tests::AttackAnswer> answers(queries.size());
    d_answers.download(answers.data(), answers.size());
    const HqAttacks& a = host_attacks();
    for (int i = 0; i < n; ++i) {
        const Square s = queries[i].square;
        const u64 occ = queries[i].occupied;
        CHECK_EQ(answers[i].rook, a.rook(s, occ));
        CHECK_EQ(answers[i].bishop, a.bishop(s, occ));
        CHECK_EQ(answers[i].queen, a.queen(s, occ));
        CHECK_EQ(answers[i].knight, a.knight(s));
        CHECK_EQ(answers[i].king, a.king(s));
        CHECK_EQ(answers[i].white_pawn, a.pawn(WHITE, s));
        CHECK_EQ(answers[i].black_pawn, a.pawn(BLACK, s));
    }

    DeviceBuffer<u64> d_between(64 * 64), d_line(64 * 64);
    gpu_tests::between_line_kernel<<<16, 256>>>(context().device_tables(), d_between.get(), d_line.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<u64> between(64 * 64), line(64 * 64);
    d_between.download(between.data(), between.size());
    d_line.download(line.data(), line.size());
    for (int i = 0; i < 64 * 64; ++i) {
        CHECK_EQ(between[i], a.between(Square(i / 64), Square(i % 64)));
        CHECK_EQ(line[i], a.line(Square(i / 64), Square(i % 64)));
    }
}

// Device code is only instantiated for the boards compiled in (CHESSGPU_BOARDS).
#if CHESSGPU_WITH_STD_BOARD
TEST_CASE(gpu_movegen_std_matches_host) { check_movegen_on_gpu<StdBoard>(); }
#endif
#if CHESSGPU_WITH_QUAD_BOARD
TEST_CASE(gpu_movegen_quad_matches_host) { check_movegen_on_gpu<QuadBoard>(); }
#endif

namespace {

void check_suite(const char* file, u64 max_nodes, BoardKind board, ExecModel model,
                 AttacksKind attacks = AttacksKind::Hq, MovesKind moves = MovesKind::Bits16) {
    std::vector<PerftCase> cases;
    std::string error;
    REQUIRE_MSG(load_perft_suite(std::string(CHESSGPU_SOURCE_DIR "/tests/perft/") + file, cases, error), error);
    GpuPerftOptions options;
    options.board = board;
    options.model = model;
    options.attacks = attacks;
    options.moves = moves;
    int checked = 0;
    for (const PerftCase& pc : cases) {
        const Position p = testutil::parse_or_die(pc.fen);
        for (const auto& [depth, expected] : pc.expected) {
            if (expected > max_nodes) break;
            const GpuPerftResult r = gpu_perft(context(), p, depth, options);
            CHECK_MSG(r.nodes == expected, std::string(name_of(board)) + "/" + name_of(model) + "/" +
                                               name_of(attacks) + "/" + name_of(moves) + "-bit perft " +
                                               std::to_string(depth) + " " + pc.fen + ": got " +
                                               std::to_string(r.nodes) + ", expected " + std::to_string(expected));
            ++checked;
        }
    }
    CHECK(checked >= int(cases.size()));
}

void check_suites(BoardKind board, ExecModel model) {
    check_suite("standard.epd", g_quick ? 200'000 : 200'000'000, board, model);
    check_suite("edge_cases.epd", g_quick ? 50'000 : 10'000'000, board, model);
}

}  // namespace

#if CHESSGPU_WITH_STD_BOARD && CHESSGPU_WITH_THREAD_MODEL
TEST_CASE(gpu_perft_suites_std_thread) { check_suites(BoardKind::Std, ExecModel::Thread); }
#endif
#if CHESSGPU_WITH_STD_BOARD && CHESSGPU_WITH_WARP_MODEL
TEST_CASE(gpu_perft_suites_std_warp) { check_suites(BoardKind::Std, ExecModel::Warp); }
#endif
#if CHESSGPU_WITH_QUAD_BOARD && CHESSGPU_WITH_THREAD_MODEL
TEST_CASE(gpu_perft_suites_quad_thread) { check_suites(BoardKind::Quad, ExecModel::Thread); }
#endif
#if CHESSGPU_WITH_QUAD_BOARD && CHESSGPU_WITH_WARP_MODEL
TEST_CASE(gpu_perft_suites_quad_warp) { check_suites(BoardKind::Quad, ExecModel::Warp); }
#endif

TEST_CASE(gpu_perft_attack_variants) {
    // Magic bitboards and Kogge-Stone fills (core/attacks_magic.h,
    // attacks_kogge.h) in both kernels and on both boards: the suites' counts.
    int runs = 0;
    for (const AttacksKind attacks : compiled_attacks()) {
        if (attacks == AttacksKind::Hq) continue;  // the suites above
        for (const BoardKind board : compiled_boards()) {
            for (const ExecModel model : compiled_models()) {
                check_suite("standard.epd", g_quick ? 200'000 : 20'000'000, board, model, attacks);
                check_suite("edge_cases.epd", g_quick ? 50'000 : 2'000'000, board, model, attacks);
                ++runs;
            }
        }
    }
    CHECK(runs > 0 || compiled_attacks().size() == 1);
}

TEST_CASE(gpu_perft_move_variants) {
    // 32-bit moves (core/move32.h) in both kernels and on both boards, with
    // hq and with magic attacks: the suites' counts.
    int runs = 0;
    for (const MovesKind moves : compiled_moves()) {
        if (moves == MovesKind::Bits16) continue;  // the suites above
        for (const AttacksKind attacks : {AttacksKind::Hq, AttacksKind::Magic}) {
            if (!is_compiled(attacks)) continue;
            for (const BoardKind board : compiled_boards()) {
                for (const ExecModel model : compiled_models()) {
                    if (attacks != AttacksKind::Hq && model != compiled_models().front()) continue;
                    check_suite("standard.epd", g_quick ? 200'000 : 20'000'000, board, model, attacks, moves);
                    check_suite("edge_cases.epd", g_quick ? 50'000 : 2'000'000, board, model, attacks, moves);
                    ++runs;
                }
            }
        }
    }
    CHECK(runs > 0 || compiled_moves().size() == 1);
}

TEST_CASE(gpu_perft_rejects_variants_not_compiled) {
    const Position start = testutil::parse_or_die(kStartFen);
    for (const BoardKind board : {BoardKind::Std, BoardKind::Quad}) {
        if (is_compiled(board)) continue;
        GpuPerftOptions options;
        options.board = board;
        options.model = compiled_models().front();
        bool threw = false;
        try {
            gpu_perft(context(), start, 2, options);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK_MSG(threw, std::string("board ") + name_of(board));
    }
    for (const ExecModel model : {ExecModel::Thread, ExecModel::Warp}) {
        if (is_compiled(model)) continue;
        GpuPerftOptions options;
        options.board = test_board();
        options.model = model;
        bool threw = false;
        try {
            gpu_perft(context(), start, 2, options);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK_MSG(threw, std::string("model ") + name_of(model));
    }
    CHECK(!compiled_boards().empty());
    CHECK(!compiled_models().empty());
}

TEST_CASE(gpu_perft_divide_matches_cpu) {
    const Position p = testutil::parse_or_die("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const int depth = g_quick ? 2 : 4;
    StdBoard b = to_board<StdBoard>(p);
    const auto expected = perft_divide(b, host_attacks(), depth);
    for (const BoardKind board : compiled_boards()) {
        for (const ExecModel model : compiled_models()) {
            GpuPerftOptions options;
            options.board = board;
            options.model = model;
            options.divide = true;
            const GpuPerftResult r = gpu_perft(context(), p, depth, options);
            CHECK(r.divide == expected);
        }
    }
}

TEST_CASE(gpu_perft_split_block_and_task_variations) {
    // The answer must not depend on how the work is split or launched.
    const Position p = testutil::parse_or_die("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const int depth = g_quick ? 3 : 4;
    const u64 expected = g_quick ? 97862 : 4085603;
    for (const ExecModel model : compiled_models()) {
        for (int split = 0; split < depth; ++split) {
            GpuPerftOptions options;
            options.board = test_board();
            options.model = model;
            options.split_depth = split;
            CHECK_MSG(gpu_perft(context(), p, depth, options).nodes == expected, "split " + std::to_string(split));
        }
        for (const int block : {32, 64, 96, 256, 512}) {
            GpuPerftOptions options;
            options.board = test_board();
            options.model = model;
            options.block_size = block;
            CHECK_MSG(gpu_perft(context(), p, depth, options).nodes == expected, "block " + std::to_string(block));
        }
        for (const int per_worker : {1, 3, 32}) {
            GpuPerftOptions options;
            options.board = test_board();
            options.model = model;
            options.tasks_per_worker = per_worker;
            CHECK_MSG(gpu_perft(context(), p, depth, options).nodes == expected,
                      "tasks per worker " + std::to_string(per_worker));
        }
    }
}

TEST_CASE(gpu_perft_edge_inputs) {
    const Position start = testutil::parse_or_die(kStartFen);
    const Position mate = testutil::parse_or_die("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
    const Position stalemate = testutil::parse_or_die("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    for (const ExecModel model : compiled_models()) {
        GpuPerftOptions options;
        options.board = test_board();
        options.model = model;
        CHECK_EQ(gpu_perft(context(), start, 0, options).nodes, u64(1));
        CHECK_EQ(gpu_perft(context(), start, 1, options).nodes, u64(20));
        CHECK_EQ(gpu_perft(context(), mate, 1, options).nodes, u64(0));
        CHECK_EQ(gpu_perft(context(), mate, 3, options).nodes, u64(0));
        CHECK_EQ(gpu_perft(context(), stalemate, 2, options).nodes, u64(0));
        options.divide = true;
        const GpuPerftResult one = gpu_perft(context(), start, 1, options);
        CHECK_EQ(one.nodes, u64(20));
        CHECK_EQ(one.divide.size(), size_t(20));
        CHECK(gpu_perft(context(), mate, 2, options).divide.empty());
    }

    bool threw = false;
    try {
        GpuPerftOptions options;
        options.board = test_board();
        options.model = compiled_models().front();
        options.block_size = 48;
        gpu_perft(context(), start, 3, options);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        GpuPerftOptions options;
        options.board = test_board();
        options.model = compiled_models().front();
        options.split_depth = 3;  // perft 3 leaves no ply for the GPU
        gpu_perft(context(), start, 3, options);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

int main(int argc, char** argv) {
    std::vector<char*> args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) g_quick = true;
        else args.push_back(argv[i]);
    }
    return test::run_all(int(args.size()), args.data());
}
