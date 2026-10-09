#include "host/host_tables.h"

#include <memory>

namespace cb {

const AttackTables& host_attack_tables() {
    // Heap-allocated: the tables are ~68 KB, too big to build on the stack.
    static const std::unique_ptr<AttackTables> tables = [] {
        auto t = std::make_unique<AttackTables>();
        init_attack_tables(*t);
        return t;
    }();
    return *tables;
}

const MagicTables& host_magic_tables() {
    static const std::unique_ptr<MagicTables> tables = [] {
        auto t = std::make_unique<MagicTables>();
        init_magic_tables(*t);
        return t;
    }();
    return *tables;
}

namespace {

SearchTables& mutable_search_tables() {
    static const std::unique_ptr<SearchTables> tables = [] {
        auto t = std::make_unique<SearchTables>();
        init_search_tables(*t);
        return t;
    }();
    return *tables;
}

}  // namespace

const SearchTables& host_search_tables() { return mutable_search_tables(); }

void set_host_nnue_weights(const NnueWeights& w) { mutable_search_tables().eval.nnue = w; }

}  // namespace cb
