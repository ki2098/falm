#include <math.h>
#include <fstream>
#include "../../src/falm.h"

using namespace Falm;

FalmCore falm;



int main(int argc, char **argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    falm.computation_init({{falm.cpm.size,1,1}}, GuideCell);

    falm.print_info();
    falm.env_finalize();
    return 0;
}