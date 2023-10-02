#include "coordinate.h"
#include "output.h"

#define L 1.0
#define N 128

using namespace std;
using namespace Falm;
using namespace LidCavity2d;

int main() {
    Matrix<REAL> x, h, kx, g, ja;
    Mapper pdm;
    setCoord(L, N, pdm, x, h, kx, g, ja);
    outputGridInfo(x, h, kx, g, ja, pdm);
}