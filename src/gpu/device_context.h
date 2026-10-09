#pragma once

#include <cstddef>
#include <mutex>
#include <string>

#include "core/attacks.h"
#include "core/attacks_magic.h"
#include "core/search_board.h"

namespace cb {

struct DeviceInfo {
    std::string name;
    int ordinal = 0;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int regs_per_sm = 0;
    size_t shared_per_sm = 0;
    size_t shared_per_block_optin = 0;
    int l2_bytes = 0;
    size_t total_memory = 0;
};

// Per-process GPU state: the selected device and the lookup tables (attack,
// Zobrist and evaluation) in device memory. Create one and pass it to the GPU
// entry points.
class DeviceContext {
  public:
    explicit DeviceContext(int device = 0);
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    const DeviceInfo& info() const { return info_; }
    const AttackTables* device_tables() const { return tables_; }                // device pointer
    const SearchTables* device_search_tables() const { return search_tables_; }  // device pointer
    // Magic bitboard tables (device pointer), searched for and copied on
    // first use (perft with AttacksKind::Magic). Thread-safe.
    const MagicTables* device_magic_tables() const;

  private:
    DeviceInfo info_;
    AttackTables* tables_ = nullptr;
    SearchTables* search_tables_ = nullptr;
    mutable std::once_flag magic_once_;
    mutable MagicTables* magic_tables_ = nullptr;
};

// Multi-line human-readable summary.
std::string describe(const DeviceInfo& info);

}  // namespace cb
