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
    printf("%u %u %u\n", u.x, u.y, u.z);
    uintx3 ux[5];
    func(ux, 5);
    for (int i = 0; i < 5; i ++) {
        printf("%u %u %u\n", ux[i].x, ux[i].y, ux[i].z);
    }

    return 0;
}