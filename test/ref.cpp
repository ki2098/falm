#include <cstdio>

struct X {
    int a;
    int b;
};

void foo(int &a, int &b) {
    a = 100;
    b = 200;
}

int main() {
    X x;
    printf("x = {%d %d}\n", x.a, x.b);
    foo(x.a, x.b);
    printf("x = {%d %d}\n", x.a, x.b);
    return 0;
}