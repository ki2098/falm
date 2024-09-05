#ifndef FALM_CPPUTIL_HPP
#define FALM_CPPUTIL_HPP

#include <string>

static std::string glue_path(const std::string &workdir, const std::string &filename) {
    if (filename[0] == '/') {
        return filename;
    } else if (workdir.back() == '/') {
        return workdir + filename;
    } else {
        return workdir + "/" + filename;
    } 
}

#endif