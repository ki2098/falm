#ifndef PROFILER_H
#define PROFILER_H

#include <sys/time.h>
#include <stdlib.h>
#include <map>

namespace Cprof {

static inline int64_t cprof_getwtime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_usec + tv.tv_sec * 1000000;
}

struct cprof_Event {
    int64_t start, duration;
};

struct cprof_Profiler {
    std::map<std::string, cprof_Event> __list;

    void startEvent(const std::string &s, bool cumulative = true) {
        __list[s].start = cprof_getwtime();
        if (!cumulative) {
            __list[s].duration = 0;
        }
    }

    void endEvent(const std::string &s) {
        __list[s].duration += cprof_getwtime() - __list[s].start;
    }

    cprof_Event &operator[](const std::string &s) {
        return __list[s];
    }
};

}

#endif