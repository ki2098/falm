#include <vector>
#include <string>
#include <stdio.h>
#include "../src/typedef.h"

struct STRUCT {
    Falm::INTx3   size;
    Falm::REALx3 coord; 
};

class Holder {
public:
    int len;
    std::string name = "holder";
    std::vector<STRUCT> structList;

    Holder(int _len) : len(_len), structList(_len, STRUCT{{0, 0, 0}, {1.0, 2.0, 3.0}}) {}
};

int main() {
    Holder h(3);
    printf("%s\n", h.name.c_str());
    for (int i = 0; i < 3; i ++) {
        STRUCT &s = h.structList[i];
        printf("%ld %ld %ld ", s.size.x, s.size.y, s.size.z);
        printf("%e %e %e\n", s.coord.x, s.coord.y, s.coord.z);
    }

    return 0;
}