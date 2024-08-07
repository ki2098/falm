#include "../src/alm/bladeHandler.h"
#include "../src/alm/apHandler.h"

int main() {
    Falm::BladeHandler::buildAP("bladeProperties.json", "apProperties.json", 2, 3, 10, 1.);

    Falm::APHandler apHandler;
    apHandler.alloc("apProperties.json");
    const Falm::APFrame &apholder = apHandler.host;

    printf("ap count %lu\n", apholder.apcount);
    printf("attack count %lu\n", apholder.attackcount);
    printf("attack list: (");
    for (int i = 0; i < apholder.attackcount; i ++) {
        printf("%lf ", apholder.attack[i]);
    }
    printf(")\n\n");

    for (int i = 0; i < apholder.apcount; i ++) {
        printf("ap id %d\n", i);
        printf("r %lf\n", apholder.r[i]);
        printf("chord %lf\n", apholder.chord[i]);
        printf("twist %lf\n", apholder.twist[i]);
        printf("Cl list: (");
        for (int j = 0; j < apholder.attackcount; j ++) {
            printf("%lf ", apholder.cl[apholder.id(i,j)]);
        }
        printf(")\n");
        printf("Cd list: (");
        for (int j = 0; j < apholder.attackcount; j ++) {
            printf("%lf ", apholder.cd[apholder.id(i,j)]);
        }
        printf(")\n\n");
    }

    apHandler.release();

    Falm::REAL3 a, b;
    a = {{0.1, 0.2, 0.3}};
    b = {{0.5, 0.6, 0.7}};
    Falm::REAL3 c = a + b;
    printf("%lf %lf %lf\n", c[0], c[1], c[2]);
}