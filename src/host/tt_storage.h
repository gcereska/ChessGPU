#pragma once

// Memory for a transposition table (core/tt.h): host memory for the CPU
// search here, device memory in gpu/device_tt.h. A table holds the largest
// power of two of 32-byte buckets that fits the requested size, at least one.

#include <cstddef>
#include <vector>

#include "core/tt.h"

namespace cb {

constexpr size_t kDefaultHashMb = 2048;
constexpr size_t kMaxHashMb = 65536;

// Buckets in a table of at most `mb` MiB (a power of two, at least 1).
u64 tt_bucket_count(size_t mb);

class TTStorage {
  public:
    virtual ~TTStorage() = default;
    virtual void resize(size_t mb) = 0;
    virtual void clear() = 0;
    // A new search: entries of earlier searches age (they lose their slots first).
    void new_search() { generation_ = (generation_ + 1) % kTTGenerations; }
    TTDesc desc() const { return TTDesc{buckets_, count_ ? count_ - 1 : 0, generation_}; }
    u64 bucket_count() const { return count_; }
    size_t size_mb() const { return size_t(count_ * sizeof(TTBucket) >> 20); }
    u32 generation() const { return generation_; }

  protected:
    TTBucket* buckets_ = nullptr;
    u64 count_ = 0;
    u32 generation_ = 0;
};

class HostTT final : public TTStorage {
  public:
    explicit HostTT(size_t mb = 16) { resize(mb); }
    void resize(size_t mb) override;
    void clear() override;

  private:
    std::vector<TTBucket> table_;
};

}  // namespace cb
