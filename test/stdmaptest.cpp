#include <sys/time.h>
#include <stdlib.h>
#include <map>
#include <unistd.h>

#include "../src/profiler.h"

int main() {
    Cprof::cprof_Profiler evl;
    for (int i = 0; i < 10; i ++) {
        evl.startEvent("test event");
        sleep(1);
        evl.endEvent("test event");
    }
    evl.startEvent("new event");
    printf("%ld\n", evl["test event"].duration);
    printf("%ld %ld\n", evl["new event"].start, evl["new event"].duration);
}