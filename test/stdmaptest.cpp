#include <sys/time.h>
#include <stdlib.h>
#include <map>
#include <unistd.h>

#include "../src/profiler.h"

int main() {
    Cprof::cprof_Profiler evl;
    for (int i = 0; i < 5; i ++) {
        std::string ename = "event " + std::to_string(i);
        evl.startEvent(ename);
        sleep(1);
        evl.endEvent(ename);
    }
    evl.output();
}