// The engine: UCI protocol handling, time management and iterative
// deepening, with the CPU backend and the GPU backends.

#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "engine/uci.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

// Lines the engine wrote, from a stream it shares with nobody while we read
// (we only read after the search finished).
std::vector<std::string> lines_of(const std::ostringstream& out) {
    std::vector<std::string> lines;
    std::istringstream in(out.str());
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

bool has_line_starting(const std::vector<std::string>& lines, const std::string& prefix) {
    for (const std::string& l : lines)
        if (l.rfind(prefix, 0) == 0) return true;
    return false;
}

std::string last_line_starting(const std::vector<std::string>& lines, const std::string& prefix) {
    std::string found;
    for (const std::string& l : lines)
        if (l.rfind(prefix, 0) == 0) found = l;
    return found;
}

// The move of the "bestmove" line, or "" if there is none.
std::string bestmove_of(const std::vector<std::string>& lines) {
    const std::string l = last_line_starting(lines, "bestmove ");
    if (l.empty()) return "";
    const size_t end = l.find(' ', 9);
    return l.substr(9, end == std::string::npos ? std::string::npos : end - 9);
}

bool is_legal_in(const std::string& fen, const std::string& move) {
    const Position p = testutil::parse_or_die(fen);
    const HqAttacks a(host_attack_tables());
    return !parse_uci_move(to_board<QuadBoard>(p), a, move).is_none();
}

int count_starting(const std::vector<std::string>& lines, const std::string& prefix) {
    int n = 0;
    for (const std::string& l : lines) n += l.rfind(prefix, 0) == 0;
    return n;
}

// The number after " <name> " in an info line (-1 if absent).
long long info_field(const std::string& line, const std::string& name) {
    const size_t at = line.find(" " + name + " ");
    if (at == std::string::npos) return -1;
    return std::stoll(line.substr(at + name.size() + 2));
}

}  // namespace

TEST_CASE(uci_handshake_and_options) {
    std::ostringstream out;
    UciEngine engine(out);
    CHECK(engine.handle("uci"));
    const std::vector<std::string> lines = lines_of(out);
    REQUIRE(!lines.empty());
    CHECK_EQ(lines.front(), std::string("id name ChessGPU"));
    CHECK_EQ(lines.back(), std::string("uciok"));
    CHECK(has_line_starting(lines, "option name Search type combo"));
    CHECK(has_line_starting(lines, "option name SplitDepth type spin"));
    CHECK(engine.handle("setoption name Search value cpu"));
    CHECK(engine.handle("isready"));
    CHECK_EQ(lines_of(out).back(), std::string("readyok"));
    // Unknown commands, options and values are reported, not fatal.
    CHECK(engine.handle("frobnicate"));
    CHECK(has_line_starting(lines_of(out), "info string unknown command: frobnicate"));
    CHECK(engine.handle("setoption name NoSuchOption value 3"));
    CHECK(has_line_starting(lines_of(out), "info string unknown option: NoSuchOption"));
    CHECK(engine.handle("setoption name Eval value bananas"));
    CHECK(has_line_starting(lines_of(out), "info string invalid value for Eval"));
    CHECK(!engine.handle("quit"));
}

TEST_CASE(uci_go_depth_cpu) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("position startpos moves e2e4 e7e5");
    engine.handle("go depth 4");
    engine.wait_for_search();
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "info string search cpu"));
    CHECK(has_line_starting(lines, "info depth 1 score cp "));
    CHECK(has_line_starting(lines, "info depth 4 score cp "));
    CHECK(!has_line_starting(lines, "info depth 5"));
    // Nothing was cut short, so no closing totals line repeats depth 4.
    CHECK_EQ(count_starting(lines, "info depth 4"), 1);
    const std::string best = bestmove_of(lines);
    CHECK_MSG(is_legal_in("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", best), best);
    // The bestmove is the first move of the last PV.
    const std::string last_info = last_line_starting(lines, "info depth 4");
    CHECK(last_info.find(" pv " + best) != std::string::npos);
}

TEST_CASE(uci_mate_and_no_moves) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
    engine.handle("go depth 3");
    engine.wait_for_search();
    std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "info depth 3 score mate 1 "));
    CHECK_EQ(bestmove_of(lines), std::string("a1a8"));
    // Mated in 1 (2 plies): "mate -1".
    engine.handle("position fen k7/8/1K6/8/8/8/8/7R b - - 0 1");
    engine.handle("go depth 3");
    engine.wait_for_search();
    lines = lines_of(out);
    CHECK(has_line_starting(lines, "info depth 3 score mate -1 "));
    // Stalemate: no legal move.
    engine.handle("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    engine.handle("go depth 2");
    engine.wait_for_search();
    CHECK_EQ(last_line_starting(lines_of(out), "bestmove"), std::string("bestmove (none)"));
}

TEST_CASE(uci_score_text) {
    CHECK_EQ(uci_score(35), std::string("cp 35"));
    CHECK_EQ(uci_score(-120), std::string("cp -120"));
    CHECK_EQ(uci_score(mate_in(1)), std::string("mate 1"));
    CHECK_EQ(uci_score(mate_in(3)), std::string("mate 2"));
    CHECK_EQ(uci_score(mate_in(4)), std::string("mate 2"));
    CHECK_EQ(uci_score(mated_in(2)), std::string("mate -1"));
    CHECK_EQ(uci_score(mated_in(4)), std::string("mate -2"));
    CHECK_EQ(uci_score(mated_in(0)), std::string("mate 0"));
}

TEST_CASE(uci_infinite_waits_for_stop) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    // Few moves: depth 63 is reached quickly, but no bestmove before "stop".
    engine.handle("position fen 8/8/8/4k3/8/8/8/4K2R w K - 0 1");
    engine.handle("go infinite");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(!has_line_starting(lines_of(out), "bestmove"));
    engine.handle("stop");
    engine.wait_for_search();
    CHECK(has_line_starting(lines_of(out), "bestmove "));
}

TEST_CASE(uci_time_limits) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("setoption name MoveOverhead value 20");
    // As a GUI does after options: the engine sets up its search (and its
    // 2 GB table) here, not on the clock.
    engine.handle("isready");
    engine.handle("position startpos");
    auto start = std::chrono::steady_clock::now();
    engine.handle("go movetime 300");
    engine.wait_for_search();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    CHECK_MSG(ms < 700, "movetime 300 took " + std::to_string(ms) + " ms");
    CHECK(!bestmove_of(lines_of(out)).empty());
    {
        // The limit (300 - 20 ms) cut an iteration short; the last info line
        // reports the totals: all the time and more nodes than any iteration.
        const std::vector<std::string> lines = lines_of(out);
        std::vector<std::string> infos;
        for (const std::string& l : lines)
            if (l.rfind("info depth ", 0) == 0) infos.push_back(l);
        REQUIRE(infos.size() >= 2);
        const std::string& last = infos.back();
        const std::string& before = infos[infos.size() - 2];
        CHECK_MSG(info_field(last, "time") >= 270, last);
        CHECK_MSG(info_field(last, "nodes") > info_field(before, "nodes"), last + " after " + before);
        CHECK_EQ(info_field(last, "depth"), info_field(before, "depth"));
        CHECK(last.find(" pv " + bestmove_of(lines)) != std::string::npos);
    }

    // A short clock: well under a second of the 2 s left.
    start = std::chrono::steady_clock::now();
    engine.handle("go wtime 2000 btime 2000 winc 0 binc 0");
    engine.wait_for_search();
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    CHECK_MSG(ms < 700, "2 s clock took " + std::to_string(ms) + " ms");

    // An exhausted clock: depth 1 only, at once.
    start = std::chrono::steady_clock::now();
    const int before = count_starting(lines_of(out), "info depth");
    engine.handle("go wtime -50 btime 1000");
    engine.wait_for_search();
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    CHECK_MSG(ms < 300, "empty clock took " + std::to_string(ms) + " ms");
    CHECK_EQ(count_starting(lines_of(out), "info depth") - before, 1);
    CHECK(!bestmove_of(lines_of(out)).empty());
}

TEST_CASE(uci_stop_right_after_go_still_moves) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("position startpos");
    engine.handle("go depth 30");
    engine.handle("stop");
    engine.wait_for_search();
    const std::string best = bestmove_of(lines_of(out));
    CHECK_MSG(is_legal_in(kStartFen, best), best);
}

TEST_CASE(uci_bad_input_keeps_state) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("position startpos moves e2e4");
    engine.handle("position fen not a fen");
    CHECK(has_line_starting(lines_of(out), "info string error: invalid FEN"));
    engine.handle("position startpos moves e2e4 e2e4");
    CHECK(has_line_starting(lines_of(out), "info string error: illegal or malformed move: e2e4"));
    engine.handle("d");
    // The last good position is kept.
    CHECK(has_line_starting(lines_of(out),
                            "info string fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"));
    engine.handle("go depth 2");
    engine.wait_for_search();
    CHECK(is_legal_in("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", bestmove_of(lines_of(out))));
}

TEST_CASE(uci_quit_during_search) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("position startpos");
    engine.handle("go infinite");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(!engine.handle("quit"));
    CHECK(has_line_starting(lines_of(out), "bestmove "));
}

namespace {

// Total nodes of the last search: the last info line before its bestmove.
long long last_nodes(const std::vector<std::string>& lines) {
    long long nodes = -1;
    for (const std::string& l : lines)
        if (l.rfind("info depth ", 0) == 0) nodes = info_field(l, "nodes");
    return nodes;
}

long long search_nodes(UciEngine& engine, std::ostringstream& out, const std::string& go) {
    out.str("");
    engine.handle(go);
    engine.wait_for_search();
    const std::vector<std::string> lines = lines_of(out);
    CHECK_MSG(!bestmove_of(lines).empty(), go);
    return last_nodes(lines);
}

}  // namespace

TEST_CASE(uci_tt_options) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines,
                            "option name Features type combo default tt-killers-see var none var tt var tt-noqs var tt-order var tt-nmp "
                            "var tt-lmr var tt-nmp-lmr var tt-fut var tt-lmr-fut var tt-ext var tt-killers "
                            "var tt-ext-killers var tt-killers-see var tt-ext-killers-see "
                            "var tt-killers-nmp var tt-killers-lmr var tt-ext-killers-nmp var tt-ext-killers-lmr "
                            "var tt-killers-see-pvs var tt-killers-see-qchecks var tt-killers-see-qchecks-nmp"));
    CHECK(has_line_starting(lines, "option name TTLayout type combo default wide128 var packed64 var wide128"));
    CHECK(has_line_starting(lines, "option name Hash type spin default 2048 min 1 max 65536"));
    CHECK(has_line_starting(lines, "option name Clear Hash type button"));
    for (const char* bad : {"setoption name Hash value 0", "setoption name Hash value 70000",
                            "setoption name Features value bananas", "setoption name TTLayout value huge"}) {
        out.str("");
        engine.handle(bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
}

TEST_CASE(uci_tt_keeps_and_clears_its_table) {
    // The CPU search with a table: deterministic, so node counts tell what
    // the table held. A second search finds the first one's entries; "Clear
    // Hash" and "ucinewgame" empty the table, and the count comes back.
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Search value cpu");
    engine.handle("setoption name Features value tt");
    engine.handle("setoption name Hash value 16");
    engine.handle("isready");
    out.str("");
    engine.handle("position startpos moves e2e4");
    const long long first = search_nodes(engine, out, "go depth 5");
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt wide128 hash 16 MB"));
    const long long again = search_nodes(engine, out, "go depth 5");
    CHECK_MSG(again < first, std::to_string(again) + " vs " + std::to_string(first));
    engine.handle("setoption name Clear Hash");
    CHECK_EQ(search_nodes(engine, out, "go depth 5"), first);
    search_nodes(engine, out, "go depth 5");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4");
    CHECK_EQ(search_nodes(engine, out, "go depth 5"), first);
    // Without the feature, no table: the plain search.
    engine.handle("setoption name Features value none");
    const long long plain = search_nodes(engine, out, "go depth 5");
    CHECK_MSG(plain > first, std::to_string(plain) + " vs " + std::to_string(first));
    // Both layouts work.
    engine.handle("setoption name Features value tt");
    engine.handle("setoption name TTLayout value packed64");
    CHECK(search_nodes(engine, out, "go depth 5") > 0);
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt packed64 hash 16 MB"));
}

TEST_CASE(uci_tt_gpu_search) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Features value tt");
    engine.handle("setoption name Hash value 64");
    engine.handle("position startpos moves d2d4 g8f6");
    const long long first = search_nodes(engine, out, "go depth 7");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "info string search ybwc-device gpu quad/nnue/thread/staged/local block 128 tt wide128 hash 64 MB"));
    CHECK(is_legal_in("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", bestmove_of(lines)));
    const long long again = search_nodes(engine, out, "go depth 7");
    CHECK_MSG(again < first, std::to_string(again) + " vs " + std::to_string(first));
}

TEST_CASE(uci_nmp_options) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "option name NullMinDepth type spin default 2 min 1 max 20"));
    CHECK(has_line_starting(lines, "option name NullReduction type spin default 3 min 1 max 10"));
    CHECK(has_line_starting(lines, "option name NullDivisor type spin default 6 min 0 max 20"));
    for (const char* bad : {"setoption name NullMinDepth value 0", "setoption name NullMinDepth value 21",
                            "setoption name NullReduction value 0", "setoption name NullReduction value 11",
                            "setoption name NullDivisor value 21", "setoption name NullDivisor value -1"}) {
        out.str("");
        engine.handle(bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
    // The settings reach the search (its description shows them).
    engine.handle("setoption name Search value cpu");
    engine.handle("setoption name Features value tt-nmp");
    engine.handle("setoption name Hash value 16");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long nodes = search_nodes(engine, out, "go depth 6");
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt-nmp wide128 null 3+d/6 from 2 hash 16 MB"));
    engine.handle("setoption name NullReduction value 2");
    engine.handle("setoption name NullDivisor value 0");
    engine.handle("setoption name NullMinDepth value 3");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long other = search_nodes(engine, out, "go depth 6");
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt-nmp wide128 null 2 from 3 hash 16 MB"));
    CHECK_MSG(other != nodes, std::to_string(other) + " vs " + std::to_string(nodes));
    // Null moves prune: the table alone searches more.
    engine.handle("setoption name Features value tt");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long tt = search_nodes(engine, out, "go depth 6");
    CHECK_MSG(tt > nodes, std::to_string(tt) + " vs " + std::to_string(nodes));
}

TEST_CASE(uci_lmr_options) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "option name LmrMinDepth type spin default 3 min 1 max 20"));
    CHECK(has_line_starting(lines, "option name LmrMoves type spin default 3 min 0 max 63"));
    CHECK(has_line_starting(lines, "option name LmrBase type spin default 75 min 0 max 255"));
    CHECK(has_line_starting(lines, "option name LmrDivisor type spin default 225 min 1 max 255"));
    CHECK(has_line_starting(lines, "option name FutDepth type spin default 3 min 1 max 10"));
    CHECK(has_line_starting(lines, "option name FutMargin type spin default 150 min 0 max 255"));
    for (const char* bad : {"setoption name LmrMinDepth value 0", "setoption name LmrMinDepth value 21",
                            "setoption name LmrMoves value 64", "setoption name LmrBase value 256",
                            "setoption name LmrDivisor value 0", "setoption name LmrDivisor value 256",
                            "setoption name FutDepth value 0", "setoption name FutDepth value 11",
                            "setoption name FutMargin value 256"}) {
        out.str("");
        engine.handle(bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
    // The settings reach the search, and reductions prune.
    engine.handle("setoption name Search value cpu");
    engine.handle("setoption name Features value tt-nmp-lmr");
    engine.handle("setoption name Hash value 16");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long lmr = search_nodes(engine, out, "go depth 7");
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt-nmp-lmr wide128 null 3+d/6 from 2 "
                                           "lmr 75/225 from 3 after 3 hash 16 MB"));
    engine.handle("setoption name LmrMoves value 10");
    engine.handle("setoption name LmrBase value 0");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long later = search_nodes(engine, out, "go depth 7");
    CHECK(has_line_starting(lines_of(out), "info string search cpu quad/nnue/staged tt-nmp-lmr wide128 null 3+d/6 from 2 "
                                           "lmr 0/225 from 3 after 10 hash 16 MB"));
    CHECK_MSG(later != lmr, std::to_string(later) + " vs " + std::to_string(lmr));
    engine.handle("setoption name Features value tt-nmp");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 e7e5");
    const long long nmp = search_nodes(engine, out, "go depth 7");
    CHECK_MSG(nmp > lmr, std::to_string(nmp) + " vs " + std::to_string(lmr));
}

TEST_CASE(uci_nmp_gpu_search) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("setoption name Features value tt-nmp");
    engine.handle("setoption name TTLayout value wide128");
    engine.handle("setoption name Hash value 64");
    engine.handle("position startpos moves d2d4 g8f6");
    search_nodes(engine, out, "go depth 7");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "info string search ybwc-device gpu quad/nnue/thread/staged/local block 128 tt-nmp "
                                   "wide128 null 3+d/6 from 2 hash 64 MB"));
    CHECK(is_legal_in("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", bestmove_of(lines)));
}

TEST_CASE(uci_gpu_search_matches_cpu) {
    // The same fixed-depth search on the CPU and on the GPU: same move.
    // Without a table: parallel searches sharing one aren't deterministic.
    const std::string fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    std::string cpu_move;
    {
        std::ostringstream out;
        UciEngine engine(out);
        engine.handle("setoption name Search value cpu");
        engine.handle("setoption name Features value none");
        engine.handle("position fen " + fen);
        engine.handle("go depth 4");
        engine.wait_for_search();
        cpu_move = bestmove_of(lines_of(out));
    }
    for (const ExecModel m : compiled_models()) {
        const char* model = name_of(m);
        std::ostringstream out;
        UciEngine engine(out);
        engine.handle("setoption name Search value ybwc-device");
        engine.handle("setoption name Features value none");
        engine.handle("setoption name LeafDepth value 1");
        engine.handle("setoption name NodeBudget value 2048");
        engine.handle("setoption name CutWaves value true");
        engine.handle(std::string("setoption name Model value ") + model);
        if (std::string(model) == "warp") engine.handle("setoption name BlockSize value 64");
        engine.handle("isready");
        engine.handle("position fen " + fen);
        engine.handle("go depth 4");
        engine.wait_for_search();
        const std::vector<std::string> lines = lines_of(out);
        CHECK_MSG(has_line_starting(lines, "info string search ybwc-device"), model);
        CHECK_MSG(bestmove_of(lines) == cpu_move, std::string(model) + " device: " + bestmove_of(lines) + " vs " + cpu_move);
    }
    for (const ExecModel m : compiled_models()) {
        const char* model = name_of(m);
        std::ostringstream out;
        UciEngine engine(out);
        engine.handle("setoption name Search value ybwc-host");
        engine.handle("setoption name Features value none");
        engine.handle("setoption name LeafDepth value 1");
        engine.handle(std::string("setoption name Model value ") + model);
        if (std::string(model) == "warp") engine.handle("setoption name BlockSize value 64");
        engine.handle("isready");
        engine.handle("position fen " + fen);
        engine.handle("go depth 4");
        engine.wait_for_search();
        const std::vector<std::string> lines = lines_of(out);
        CHECK_MSG(has_line_starting(lines, "info string search ybwc-host"), model);
        CHECK_MSG(bestmove_of(lines) == cpu_move, std::string(model) + " ybwc: " + bestmove_of(lines) + " vs " + cpu_move);
    }
    for (const ExecModel m : compiled_models()) {
        const char* model = name_of(m);
        std::ostringstream out;
        UciEngine engine(out);
        engine.handle("setoption name Search value gpu-independent");
        engine.handle("setoption name Features value none");
        engine.handle(std::string("setoption name Model value ") + model);
        if (std::string(model) == "warp") engine.handle("setoption name BlockSize value 64");
        engine.handle("isready");
        engine.handle("position fen " + fen);
        engine.handle("go depth 4");
        engine.wait_for_search();
        const std::vector<std::string> lines = lines_of(out);
        CHECK_MSG(has_line_starting(lines, "info string search gpu-independent"), model);
        CHECK_MSG(bestmove_of(lines) == cpu_move, std::string(model) + ": " + bestmove_of(lines) + " vs " + cpu_move);
    }
}

TEST_CASE(uci_attacks_option) {
    // The search kernels' sliding attacks: hq (the default) or magic; the
    // perft-only Kogge-Stone isn't offered.
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    CHECK(has_line_starting(lines_of(out), "option name Attacks type combo default hq var hq var magic"));
    out.str("");
    engine.handle("setoption name Attacks value kogge");
    CHECK(has_line_starting(lines_of(out), "info string invalid value"));
    engine.handle("setoption name Attacks value magic");
    engine.handle("setoption name Hash value 64");
    engine.handle("position startpos moves e2e4");
    CHECK(search_nodes(engine, out, "go depth 6") > 0);
    CHECK(has_line_starting(lines_of(out), "info string search ybwc-device gpu quad/nnue/thread/staged/local block 128 "
                                           "magic tt-killers-see wide128 delta 200 see hash 64 MB"));
}

TEST_CASE(uci_see_options) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "option name DeltaMargin type spin default 200 min 0 max 2000"));
    CHECK(has_line_starting(lines, "option name SeePrune type check default true"));
    for (const char* bad : {"setoption name DeltaMargin value 2001", "setoption name SeePrune value maybe"}) {
        out.str("");
        engine.handle(bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
    // The settings reach the search; with both prunings off the search is
    // tt-killers' exactly, and with them it searches fewer nodes.
    engine.handle("setoption name Search value cpu");
    engine.handle("setoption name Hash value 16");
    engine.handle("setoption name Features value tt-killers-see");
    engine.handle("position startpos moves e2e4 d7d5");
    const long long pruned = search_nodes(engine, out, "go depth 6");
    CHECK(has_line_starting(lines_of(out),
                            "info string search cpu quad/nnue/staged tt-killers-see wide128 delta 200 see hash 16 MB"));
    engine.handle("setoption name DeltaMargin value 0");
    engine.handle("setoption name SeePrune value false");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 d7d5");
    const long long off = search_nodes(engine, out, "go depth 6");
    CHECK(has_line_starting(lines_of(out),
                            "info string search cpu quad/nnue/staged tt-killers-see wide128 no delta no see hash 16 MB"));
    engine.handle("setoption name Features value tt-killers");
    engine.handle("ucinewgame");
    engine.handle("position startpos moves e2e4 d7d5");
    const long long plain = search_nodes(engine, out, "go depth 6");
    CHECK_MSG(off == plain && pruned < plain,
              std::to_string(pruned) + " pruned, " + std::to_string(off) + " off, " + std::to_string(plain) + " plain");
}

TEST_CASE(uci_abdada_searches) {
    // ABDADA on CPU threads and on the GPU: the options reach the backends,
    // and both play legal moves.
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "option name Threads type spin default 4 min 1 max 256"));
    CHECK(has_line_starting(lines, "option name Workers type spin default 2048 min 1 max 1048576"));
    CHECK(has_line_starting(lines, "option name AbdadaMinDepth type spin default 2 min 1 max 20"));
    CHECK(has_line_starting(lines, "option name AbdadaDeferFirst type check default false"));
    for (const char* bad : {"setoption name Workers value 0", "setoption name Threads value 257",
                            "setoption name AbdadaMinDepth value 0", "setoption name AbdadaDeferFirst value yes"}) {
        out.str("");
        engine.handle(bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
    const std::string fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8";
    engine.handle("setoption name Hash value 64");
    engine.handle("setoption name Search value abdada-cpu");
    engine.handle("setoption name Threads value 3");
    engine.handle("position fen " + fen);
    CHECK(search_nodes(engine, out, "go depth 6") > 0);
    CHECK(has_line_starting(lines_of(out),
                            "info string search abdada-cpu 3 workers quad/nnue/list tt-killers-see wide128 delta 200 see "
                            "hash 64 MB"));
    CHECK(is_legal_in(fen, bestmove_of(lines_of(out))));
    engine.handle("setoption name Search value abdada-device");
    engine.handle("setoption name Workers value 512");
    engine.handle("ucinewgame");
    engine.handle("position fen " + fen);
    CHECK(search_nodes(engine, out, "go depth 6") > 0);
    CHECK(has_line_starting(lines_of(out), "info string search abdada-device gpu quad/nnue/thread/list/local block 128 "
                                           "tt-killers-see wide128 delta 200 see hash 64 MB workers 512 lanes 1"));
    CHECK(is_legal_in(fen, bestmove_of(lines_of(out))));
    // The coordination settings reach both searches.
    engine.handle("setoption name AbdadaMinDepth value 3");
    engine.handle("setoption name AbdadaDeferFirst value true");
    engine.handle("position fen " + fen);
    CHECK(search_nodes(engine, out, "go depth 6") > 0);
    CHECK(has_line_starting(lines_of(out), "info string search abdada-device gpu quad/nnue/thread/list/local block 128 "
                                           "tt-killers-see wide128 delta 200 see hash 64 MB workers 512 lanes 1 "
                                           "from depth 3 defer first"));
    CHECK(is_legal_in(fen, bestmove_of(lines_of(out))));
    engine.handle("setoption name Search value abdada-cpu");
    engine.handle("position fen " + fen);
    CHECK(search_nodes(engine, out, "go depth 5") > 0);
    CHECK(has_line_starting(lines_of(out), "info string search abdada-cpu 3 workers quad/nnue/list tt-killers-see "
                                           "wide128 delta 200 see hash 64 MB from depth 3 defer first"));
    CHECK(is_legal_in(fen, bestmove_of(lines_of(out))));
}


TEST_CASE(uci_aspiration_option) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    CHECK(has_line_starting(lines_of(out), "option name Aspiration type spin default 0 min 0 max 1000"));
    CHECK(has_line_starting(lines_of(out), "option name FailLowTime type spin default 0 min 0 max 400"));
    out.str("");
    engine.handle("setoption name Aspiration value 1001");
    CHECK(has_line_starting(lines_of(out), "info string invalid value"));
    out.str("");
    engine.handle("setoption name FailLowTime value 401");
    CHECK(has_line_starting(lines_of(out), "info string invalid value"));
    out.str("");
    engine.handle("setoption name FailLowTime value 100");
    CHECK(!has_line_starting(lines_of(out), "info string invalid value"));
    // Narrow windows on the CPU and on the GPU: legal moves.
    const std::string fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8";
    engine.handle("setoption name Aspiration value 5");
    engine.handle("setoption name Hash value 64");
    for (const char* search : {"cpu", "ybwc-device"}) {
        engine.handle(std::string("setoption name Search value ") + search);
        engine.handle("ucinewgame");
        engine.handle("position fen " + fen);
        CHECK_MSG(search_nodes(engine, out, "go depth 7") > 0, search);
        CHECK_MSG(is_legal_in(fen, bestmove_of(lines_of(out))), search);
        // On a clock, with the fail-low extension on (FailLowTime 100 above).
        CHECK_MSG(search_nodes(engine, out, "go wtime 3000 btime 3000 winc 100 binc 100") > 0, search);
        CHECK_MSG(is_legal_in(fen, bestmove_of(lines_of(out))), search);
    }
}

namespace {

// The "info depth <depth> multipv <k>" lines of a search, by k (1-based).
std::vector<std::string> multipv_lines(const std::vector<std::string>& lines, int depth) {
    std::vector<std::string> out;
    for (const std::string& l : lines) {
        if (l.rfind("info depth " + std::to_string(depth) + " multipv ", 0) != 0) continue;
        const long long k = info_field(l, "multipv");
        if (k >= 1 && size_t(k) > out.size()) out.resize(size_t(k));
        if (k >= 1) out[size_t(k - 1)] = l;
    }
    return out;
}

std::string first_pv_move(const std::string& line) {
    const size_t at = line.find(" pv ");
    if (at == std::string::npos) return "";
    const size_t end = line.find(' ', at + 4);
    return line.substr(at + 4, end == std::string::npos ? std::string::npos : end - at - 4);
}

}  // namespace

TEST_CASE(uci_multipv) {
    std::ostringstream out;
    UciEngine engine(out);
    engine.handle("uci");
    CHECK(has_line_starting(lines_of(out), "option name MultiPV type spin default 1 min 1 max 256"));
    for (const char* bad : {"0", "257", "two"}) {
        out.str("");
        engine.handle(std::string("setoption name MultiPV value ") + bad);
        CHECK_MSG(has_line_starting(lines_of(out), "info string invalid value"), bad);
    }
    // The CPU and the GPU (YBWC on the device), both without a table (exact
    // scores): three lines per depth, best first, with the same scores.
    const std::string fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    std::vector<long long> cpu_scores;
    for (const char* search : {"cpu", "ybwc-device"}) {
        engine.handle(std::string("setoption name Search value ") + search);
        engine.handle("setoption name Features value none");
        engine.handle("setoption name MultiPV value 3");
        engine.handle("ucinewgame");
        engine.handle("position fen " + fen);
        out.str("");
        engine.handle("go depth 4");
        engine.wait_for_search();
        const std::vector<std::string> lines = lines_of(out);
        for (int depth = 1; depth <= 4; ++depth) {
            const std::vector<std::string> ranked = multipv_lines(lines, depth);
            REQUIRE_MSG(ranked.size() == 3, std::string(search) + " depth " + std::to_string(depth));
            std::vector<long long> scores;
            for (size_t k = 0; k < ranked.size(); ++k) {
                REQUIRE_MSG(!ranked[k].empty(), std::string(search) + " line " + std::to_string(k + 1));
                scores.push_back(info_field(ranked[k], "cp"));
                CHECK(is_legal_in(fen, first_pv_move(ranked[k])));
                if (k > 0) {
                    CHECK(scores[k] <= scores[k - 1]);
                    CHECK(first_pv_move(ranked[k]) != first_pv_move(ranked[k - 1]));
                }
            }
            if (depth == 4) {
                CHECK(bestmove_of(lines) == first_pv_move(ranked[0]));
                if (cpu_scores.empty()) cpu_scores = scores;
                else
                    CHECK_MSG(scores == cpu_scores, std::string(search) + ": " + std::to_string(scores[0]) + " " +
                                                        std::to_string(scores[1]) + " " + std::to_string(scores[2]) +
                                                        " vs " + std::to_string(cpu_scores[0]) + " " +
                                                        std::to_string(cpu_scores[1]) + " " +
                                                        std::to_string(cpu_scores[2]));
            }
        }
    }
    // Back to one line: plain info lines.
    engine.handle("setoption name MultiPV value 1");
    engine.handle("setoption name Search value cpu");
    engine.handle("position fen " + fen);
    out.str("");
    engine.handle("go depth 3");
    engine.wait_for_search();
    const std::vector<std::string> lines = lines_of(out);
    CHECK(has_line_starting(lines, "info depth 3 score ") && !has_line_starting(lines, "info depth 3 multipv"));
}
