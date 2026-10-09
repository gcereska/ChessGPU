// Self-play for tuning data (host/selfplay.h): a few short games on two
// threads write well-formed "<FEN> | <result> | <game>" lines, every game
// numbered once with one result for all its positions, and a game that is
// already decided ends by adjudication or mate with the right winner.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>

#include "host/fen.h"
#include "host/selfplay.h"
#include "test_harness.h"

using namespace cb;

namespace {

std::string temp_file(const char* name) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::error_code ignored;
    std::filesystem::remove(p, ignored);
    return p.string();
}

}  // namespace

TEST_CASE(selfplay_writes_games) {
    SelfPlayOptions o;
    o.games = 6;
    o.depth = 2;
    o.threads = 2;
    o.openings = {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                  "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"};
    o.out_path = temp_file("chessgpu_selfplay_test.txt");
    o.search.features = FeatureKind::TTKillers;
    SelfPlayStats stats;
    try {
        stats = run_selfplay(o, {});
    } catch (const std::exception& e) {
        CHECK_MSG(false, e.what());
        return;
    }
    CHECK(stats.games >= 1 && stats.games <= 6);
    CHECK_EQ(stats.white_wins + stats.black_wins + stats.draws, stats.games);
    std::ifstream in(o.out_path);
    std::map<int, std::string> result_of;
    unsigned long long lines = 0;
    int nonzero_scores = 0;
    for (std::string line; std::getline(in, line);) {
        // "<FEN> | <result> | <game> | <white's score>"
        const size_t a = line.find(" | ");
        const size_t b = a == std::string::npos ? a : line.find(" | ", a + 3);
        const size_t c = line.rfind(" | ");
        REQUIRE_MSG(a != std::string::npos && b != std::string::npos && c != b, line);
        const int score = std::stoi(line.substr(c + 3));
        CHECK_MSG(score > -kScoreInfinite && score < kScoreInfinite, line);
        nonzero_scores += score != 0 ? 1 : 0;
        Position p;
        std::string error;
        CHECK_MSG(parse_fen(line.substr(0, a), p, error), line);
        const std::string result = line.substr(a + 3, b - a - 3);
        CHECK_MSG(result == "1" || result == "0" || result == "0.5", line);
        const int game = std::stoi(line.substr(b + 3, c - b - 3));
        CHECK(game >= 1 && game <= stats.games);
        if (result_of.count(game)) CHECK_MSG(result_of[game] == result, line);
        result_of[game] = result;
        ++lines;
    }
    CHECK_EQ(lines, stats.positions);
    CHECK_EQ(int(result_of.size()), stats.games);
    CHECK(nonzero_scores * 2 > int(lines));  // the scores are real search scores
    in.close();  // Windows can't remove an open file
    std::error_code ignored;
    std::filesystem::remove(o.out_path, ignored);
}

TEST_CASE(selfplay_decided_games_end_right) {
    // White has a queen and rook against a bare king: white wins, by mate or
    // by adjudication, in every game.
    SelfPlayOptions o;
    o.games = 3;
    o.depth = 3;
    o.threads = 1;
    o.random_plies = 0;
    o.openings = {"k7/8/8/8/8/8/8/1QR3K1 w - - 0 1"};
    o.out_path = temp_file("chessgpu_selfplay_decided.txt");
    o.search.features = FeatureKind::TTKillers;
    SelfPlayStats stats;
    try {
        stats = run_selfplay(o, {});
    } catch (const std::exception& e) {
        CHECK_MSG(false, e.what());
        return;
    }
    CHECK_EQ(stats.games, 3);
    CHECK_EQ(stats.white_wins, 3);
    std::error_code ignored;
    std::filesystem::remove(o.out_path, ignored);
}
