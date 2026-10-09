#pragma once

// The UCI protocol (https://www.chessprogramming.org/UCI): commands come in
// one line at a time; searches run on their own thread, so "stop" and
// "isready" are answered while one runs.

#include <condition_variable>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include "engine/backends.h"

namespace cb {

class UciEngine {
  public:
    explicit UciEngine(std::ostream& out);
    ~UciEngine();
    UciEngine(const UciEngine&) = delete;
    UciEngine& operator=(const UciEngine&) = delete;

    // Handles one line from the GUI. Returns false after "quit".
    bool handle(const std::string& line);

    // Blocks until the running search, if any, has sent its bestmove.
    void wait_for_search();

  private:
    void send(const std::string& line);
    void uci();
    void is_ready();
    void set_option(const std::vector<std::string>& tokens);
    void position(const std::vector<std::string>& tokens);
    void go(const std::vector<std::string>& tokens);
    void stop();
    void ensure_backend();
    void finish_search();

    std::ostream& out_;
    std::mutex out_mutex_;

    EngineOptions options_;
    double move_overhead_ms_ = 50;
    int multipv_ = 1;  // UCI MultiPV: lines per iteration (a search setting: the backend stays)
    std::unique_ptr<DeviceContext> ctx_;
    std::unique_ptr<SearchBackend> backend_;
    bool backend_stale_ = true;

    Position root_;
    GameHistory history_;

    std::thread search_thread_;
    std::mutex search_mutex_;
    std::condition_variable search_cv_;
    bool searching_ = false;      // guarded by search_mutex_
    bool stop_received_ = false;  // guarded by search_mutex_
};

// UCI score text: "cp 35", "mate 3" (we mate in 3 moves), "mate -2" (we are
// mated in 2) or "mate 0" (checkmated now).
std::string uci_score(int score);

}  // namespace cb
