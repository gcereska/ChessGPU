#pragma once

// A transposition table in device memory, shared by every GPU worker of a
// search and by the GPU-resident YBWC's explicit nodes (core/tt.h).

#include "host/tt_storage.h"

namespace cb {

class DeviceTT final : public TTStorage {
  public:
    explicit DeviceTT(size_t mb = kDefaultHashMb) { resize(mb); }
    ~DeviceTT() override;
    DeviceTT(const DeviceTT&) = delete;
    DeviceTT& operator=(const DeviceTT&) = delete;

    void resize(size_t mb) override;  // allocates and clears
    void clear() override;            // waits until the table is cleared
};

}  // namespace cb
