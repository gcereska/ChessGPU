#pragma once

// Minimal self-contained test framework.
//
//   TEST_CASE(name) { CHECK(x); CHECK_EQ(a, b); REQUIRE(y); }
//
// CHECK/CHECK_EQ record a failure and continue; REQUIRE returns from the test.
// Run with an optional substring filter: unit_tests.exe attacks

#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct TestState {
    int failures = 0;  // failed checks in the current test
};

inline TestState& current() {
    static TestState state;
    return state;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

template <class T>
std::string show(const T& value) {
    std::ostringstream os;
    if constexpr (std::is_enum_v<T>) {
        os << static_cast<long long>(value);
    } else if constexpr (std::is_same_v<T, unsigned char> || std::is_same_v<T, signed char>) {
        os << static_cast<int>(value);
    } else {
        os << value;
    }
    return os.str();
}

// Prints at most this many failures per test so a broken loop doesn't flood the log.
constexpr int kMaxPrintedFailures = 20;

inline void report_failure(const char* file, int line, const std::string& message) {
    if (current().failures < kMaxPrintedFailures) std::printf("    FAILED %s:%d: %s\n", file, line, message.c_str());
    if (current().failures == kMaxPrintedFailures) std::printf("    (further failures in this test not printed)\n");
    ++current().failures;
}

inline int run_all(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int run = 0;
    int failed = 0;
    for (const TestCase& t : registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        ++run;
        current().failures = 0;
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        const auto start = std::chrono::steady_clock::now();
        t.fn();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (current().failures) {
            ++failed;
            std::printf("[ FAIL ] %s (%d failed checks, %.1f ms)\n", t.name, current().failures, ms);
        } else {
            std::printf("[  OK  ] %s (%.1f ms)\n", t.name, ms);
        }
        std::fflush(stdout);
    }
    std::printf("\n%d tests run, %d failed\n", run, failed);
    if (run == 0) {
        std::printf("no tests matched filter \"%s\"\n", filter.c_str());
        return 1;
    }
    return failed ? 1 : 0;
}

}  // namespace test

#define TEST_CASE(name)                                                \
    static void name();                                                \
    static const ::test::Registrar name##_registrar(#name, &name);     \
    static void name()

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) ::test::report_failure(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

// Copies both values: binding a reference would dangle when (a) refers into a
// temporary, e.g. CHECK_EQ(make_vector().back(), x).
#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const auto check_a_ = (a);                                                                  \
        const auto check_b_ = (b);                                                                  \
        if (!(check_a_ == check_b_))                                                                \
            ::test::report_failure(__FILE__, __LINE__,                                              \
                                   "CHECK_EQ(" #a ", " #b "): " + ::test::show(check_a_) + " vs " + \
                                       ::test::show(check_b_));                                     \
    } while (0)

// Like CHECK, plus a message (any expression convertible to std::string).
#define CHECK_MSG(cond, msg)                                                                              \
    do {                                                                                                  \
        if (!(cond)) ::test::report_failure(__FILE__, __LINE__, std::string("CHECK(" #cond "): ") + (msg)); \
    } while (0)

#define REQUIRE(cond)                                                                    \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            ::test::report_failure(__FILE__, __LINE__, "REQUIRE(" #cond ")");            \
            return;                                                                      \
        }                                                                                \
    } while (0)

#define REQUIRE_MSG(cond, msg)                                                                         \
    do {                                                                                               \
        if (!(cond)) {                                                                                 \
            ::test::report_failure(__FILE__, __LINE__, std::string("REQUIRE(" #cond "): ") + (msg));   \
            return;                                                                                    \
        }                                                                                              \
    } while (0)
