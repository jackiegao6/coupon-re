#pragma once
#include <cstdint>
#include <stdexcept>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

struct Memory {
    int64_t peak_private;
    uint64_t peak_working_set;
    int64_t current_private;
};

inline Memory memory_usage() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX p{};
    p.cb = sizeof(p);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&p), sizeof(p)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    return {int64_t(p.PeakPagefileUsage), uint64_t(p.PeakWorkingSetSize), int64_t(p.PrivateUsage)};
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error("getrusage failed");
#if defined(__APPLE__)
    uint64_t rss = uint64_t(usage.ru_maxrss);
#else
    uint64_t rss = uint64_t(usage.ru_maxrss) * 1024;
#endif
    // Private commit is not the same metric as POSIX peak RSS.
    return {-1, rss, -1};
#endif
}
