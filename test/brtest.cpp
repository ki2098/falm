#include <stdio.h>

int main() {
    int i;
    for (i = 0; i < 1000; i ++) {
        if (i == 100) break;
    }
    printf("%d\n", i);
}