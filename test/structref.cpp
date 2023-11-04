#include <stdio.h>

struct uintx3 {
    unsigned int x, y, z;
};

void func(uintx3 &u) {
    u = {1, 2, 3};
}

void func(uintx3 *uptr, int n) {
    for (unsigned int i = 0; i < n; i ++) {
        uptr[i] = {i * 3, i * 3 + 1, i * 3 + 2};
    }
}

int main() {
    uintx3 u{0, 0, 0};
    func(u);
    printf("%u %u %u\n", u[0], u[1], u[2]);
    uintx3 ux[5];
    func(ux, 5);
    for (int i = 0; i < 5; i ++) {
        printf("%u %u %u\n", ux[i][0], ux[i][1], ux[i][2]);
    }

    return 0;
}