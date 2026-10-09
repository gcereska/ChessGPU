#include "host/tt_storage.h"

#include <algorithm>

namespace cb {

u64 tt_bucket_count(size_t mb) {
    const u64 bytes = u64(std::max<size_t>(mb, 1)) << 20;
    u64 count = 1;
    while (count * 2 * sizeof(TTBucket) <= bytes) count *= 2;
    return count;
}

void HostTT::resize(size_t mb) {
    const u64 count = tt_bucket_count(mb);
    table_.assign(size_t(count), TTBucket{});
    buckets_ = table_.data();
    count_ = count;
}

void HostTT::clear() { std::fill(table_.begin(), table_.end(), TTBucket{}); }

}  // namespace cb
