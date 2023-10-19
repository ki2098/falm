#include "__vcdm.h"

std::string __VCDM::makefilename(const std::string &prefix) {
    return vcdm.makeFilename(Vcdm::FilenameFormat::RANK, prefix, "", "", 0, 0);
}