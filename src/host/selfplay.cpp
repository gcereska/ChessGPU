#include "host/selfplay.h"

#include <atomic>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/movegen.h"
#include "core/search_board.h"
#include "core/zobrist.h"
#include "host/fen.h"
#include "host/game_history.h"
#include "host/host_tables.h"
#include "host/search_backend.h"
#include "host/uci_move.h"

namespace cb {

namespace {

enum class Outcome { Running, WhiteWins, BlackWins, Draw };

const char* result_text(Outcome o) { return o == Outcome::WhiteWins ? "1" : o == Outcome::BlackWins ? "0" : "0.5"; }

struct Game {
    Position start;
    std::string moves;          // UCI, from start
    Position now;
    std::vector<u64> keys;      // positions since the last capture or pawn move, oldest first
    std::vector<std::string> fens;  // positions recorded for the tuning set
    std::vector<int> scores;        // the search's score of each, white's side
};

// Plays m: updates the position, the move list and the repetition keys.
void play(Game& g, Move m) {
    const HqAttacks a(host_attack_tables());
    QuadBoard b = to_board<QuadBoard>(g.now);
    const bool irreversible = m.is_capture() || b.piece_type_on(m.from()) == PAWN;
    make_move(b, m);
    g.now = to_position(b, g.now.fullmove + (state::side(g.now.state) == BLACK ? 1 : 0));
    g.moves += (g.moves.empty() ? "" : " ") + move_to_uci(m);
    if (irreversible) g.keys.clear();
    g.keys.push_back(compute_key(b, a, host_search_tables().zobrist));
}

// The game's end by the rules, if it has ended.
Outcome rules_outcome(const Game& g) {
    const HqAttacks a(host_attack_tables());
    const QuadBoard b = to_board<QuadBoard>(g.now);
    if (count_legal(b, a) == 0) {
        if (!side_to_move_in_check(b, a)) return Outcome::Draw;  // stalemate
        return state::side(g.now.state) == WHITE ? Outcome::BlackWins : Outcome::WhiteWins;
    }
    if (state::halfmove(g.now.state) >= 100 || insufficient_material(b)) return Outcome::Draw;
    int same = 0;
    for (const u64 k : g.keys) same += k == g.keys.back();
    return same >= 3 ? Outcome::Draw : Outcome::Running;
}

Outcome play_game(const SelfPlayOptions& o, CpuBackend& backend, std::mt19937_64& rng, Game& g) {
    const HqAttacks a(host_attack_tables());
    std::string error;
    const std::string& fen = o.openings.empty() ? std::string(kStartFen) : o.openings[rng() % o.openings.size()];
    if (!parse_fen(fen, g.start, error)) throw std::runtime_error("bad opening " + fen + ": " + error);
    g.now = g.start;
    g.keys = {compute_key(to_board<QuadBoard>(g.now), a, host_search_tables().zobrist)};
    for (int i = 0; i < o.random_plies; ++i) {
        if (rules_outcome(g) != Outcome::Running) return Outcome::Running;  // not a game worth keeping
        MoveList list;
        generate_legal(to_board<QuadBoard>(g.now), a, list);
        play(g, list.moves[rng() % unsigned(list.size)]);
    }
    backend.clear_hash();
    int lopsided = 0;  // consecutive plies with |score| >= adjudicate_cp, agreeing on the winner
    int last_sign = 0;
    for (int ply = 0; ply < o.max_plies; ++ply) {
        const Outcome rules = rules_outcome(g);
        if (rules != Outcome::Running) return rules;
        g.fens.push_back(format_fen(g.now));
        SearchRequest request;
        if (!build_game(g.start, g.moves, request.root, request.history, error)) throw std::runtime_error(error);
        request.max_depth = o.depth;
        backend.new_search();
        const SearchOutcome s = iterative_deepening(backend, request, {});
        if (s.best.is_none()) throw std::runtime_error("no move in " + format_fen(g.now));
        const int white_score = state::side(g.now.state) == WHITE ? s.score : -s.score;
        g.scores.push_back(white_score);
        const int sign = white_score >= o.adjudicate_cp ? 1 : (white_score <= -o.adjudicate_cp ? -1 : 0);
        lopsided = (sign != 0 && sign == last_sign) ? lopsided + 1 : (sign != 0 ? 1 : 0);
        last_sign = sign;
        if (lopsided >= 4) return sign > 0 ? Outcome::WhiteWins : Outcome::BlackWins;
        play(g, s.best);
    }
    return Outcome::Draw;
}

}  // namespace

SelfPlayStats run_selfplay(const SelfPlayOptions& o, const std::function<void(const SelfPlayStats&)>& progress) {
    if (o.games < 1 || o.depth < 1 || o.threads < 1) throw std::invalid_argument("bad self-play options");
    std::ofstream out(o.out_path, std::ios::app);
    if (!out) throw std::runtime_error("can't write " + o.out_path);
    std::mutex mutex;  // guards out, stats and the progress callback
    SelfPlayStats stats;
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::string failure;
    const auto worker = [&](int thread) {
        try {
            CpuBackend backend(o.search, o.hash_mb);
            std::mt19937_64 rng(o.seed * 1000003ull + unsigned(thread));
            for (int i = next++; i < o.games && !failed; i = next++) {
                Game g;
                const Outcome outcome = play_game(o, backend, rng, g);
                if (outcome == Outcome::Running) continue;  // ended during the random moves
                std::lock_guard<std::mutex> lock(mutex);
                const int number = o.first_game + stats.games;
                for (size_t k = 0; k < g.fens.size(); ++k)
                    out << g.fens[k] << " | " << result_text(outcome) << " | " << number << " | " << g.scores[k]
                        << '\n';
                ++stats.games;
                stats.positions += g.fens.size();
                if (outcome == Outcome::WhiteWins) ++stats.white_wins;
                else if (outcome == Outcome::BlackWins) ++stats.black_wins;
                else ++stats.draws;
                if (progress) progress(stats);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex);
            failure = e.what();
            failed = true;
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < o.threads; ++t) threads.emplace_back(worker, t);
    for (std::thread& t : threads) t.join();
    if (failed) throw std::runtime_error("self-play failed: " + failure);
    return stats;
}

}  // namespace cb
