#include <vector>
#include <string>
#include <stdio.h>
#include "../src/typedef.h"

struct STRUCT {
    Falm::Int3   size;
    Falm::Real3 coord; 
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
        printf("%ld %ld %ld ", s.size[0], s.size[1], s.size[2]);
        printf("%e %e %e\n", s.coord[0], s.coord[1], s.coord[2]);
    }

    return 0;
}