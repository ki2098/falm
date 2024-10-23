#include "../src/alm/bladeHandler.h"

int main() {
    std::ifstream bladeperp("bladeProperties.json");
    auto bladejson = Falm::json::parse(bladeperp);

    Falm::Alm::BladeHandler::buildAP(bladejson, "apProperties.json", 2, 3, 5, 1.0, 0.07);
}