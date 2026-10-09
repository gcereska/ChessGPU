#include "host/epd.h"

#include <charconv>
#include <fstream>

namespace cb {

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

template <class T>
bool parse_number(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc() && end == text.data() + text.size();
}

}  // namespace

bool load_perft_suite(const std::string& path, std::vector<PerftCase>& out, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<PerftCase> cases;
    std::string raw;
    int line_number = 0;
    while (std::getline(in, raw)) {
        ++line_number;
        const std::string_view line = trim(raw);
        if (line.empty() || line.front() == '#') continue;

        const auto where = [&] { return path + ":" + std::to_string(line_number) + ": "; };
        PerftCase pc;
        pc.line = line_number;
        size_t semicolon = line.find(';');
        pc.fen = std::string(trim(line.substr(0, semicolon)));
        while (semicolon != std::string_view::npos) {
            const size_t next = line.find(';', semicolon + 1);
            const std::string_view entry = trim(line.substr(semicolon + 1, next - semicolon - 1));
            semicolon = next;
            // Expect "D<depth> <nodes>".
            const size_t space = entry.find(' ');
            int depth = 0;
            u64 nodes = 0;
            if (entry.size() < 2 || entry[0] != 'D' || space == std::string_view::npos ||
                !parse_number(entry.substr(1, space - 1), depth) || !parse_number(trim(entry.substr(space)), nodes) ||
                depth < 1) {
                error = where() + "expected \"D<depth> <nodes>\", got \"" + std::string(entry) + "\"";
                return false;
            }
            if (!pc.expected.empty() && depth <= pc.expected.back().first) {
                error = where() + "depths must increase";
                return false;
            }
            pc.expected.emplace_back(depth, nodes);
        }
        if (pc.expected.empty()) {
            error = where() + "no expected counts";
            return false;
        }
        cases.push_back(std::move(pc));
    }
    out = std::move(cases);
    return true;
}

}  // namespace cb
