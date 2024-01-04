#include "region.h"
#include "devdefine.h"

namespace Falm {

class CpmBuffer {
public:
    void *dptr = nullptr;
    void *hptr = nullptr;
    bool active = false;
    Region map;
    int count = 0;
    size_t capacity = 0;
    size_t width = 0;
    INT color;
    FLAG hdc;
};

class CpmBufMan {
public:
    static const int NBUFFER = 108;
public:
    int max_used_idx = 0;
    CpmBuffer buffer[NBUFFER];
    FLAG hdc;
    int request(size_t width, const Region &map, const Region &pdm, INT color) {
        INT refcolor = (SUM3(pdm.offset) + SUM3(map.offset)) % 2;
        int count = map.size / 2;
        if (map.size % 2 == 1 && refcolor == color) {
            count ++;
        }
        size_t reqsize = width * count;
        int first_vacant = -1;
        for (int i = 0; i <= max_used_idx; i ++) {
            if (!buffer[i].active) {
                if (first_vacant == -1) {
                    first_vacant = i;
                }
            }
        }
    }
};

}