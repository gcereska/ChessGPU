#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/types.h"

namespace cb {

// One line of a perft suite: a FEN and the expected node count per depth.
struct PerftCase {
    std::string fen;
    std::vector<std::pair<int, u64>> expected;  // (depth, nodes), increasing depth
    int line = 0;                               // line number in the file, for messages
};

// Reads a perft suite: one position per line, "FEN ;D1 20 ;D2 400 ...".
// Blank lines and lines starting with '#' are skipped. FENs are not
// validated here. On failure returns false and sets `error`.
bool load_perft_suite(const std::string& path, std::vector<PerftCase>& out, std::string& error);

}  // namespace cb
