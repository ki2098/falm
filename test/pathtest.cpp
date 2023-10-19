#include <string>
#include <cstdio>
#include <iostream>

using namespace std;

std::string cut_filename(const std::string &path) {
    if (path.find_last_of('/') == std::string::npos) {
        return path;
    } else {
        return path.substr(path.find_last_of('/') + 1);
    }
}

std::string cut_dirpath(const std::string &path) {
    if (path.find_first_of('/') == std::string::npos) {
        return "";
    } else {
        return path.substr(0, path.find_last_of('/'));
    }
}

int main(int argc, char **argv) {
    string path(argv[1]);
    // cin >> path;
    string filename = cut_filename(path);
    string dir = cut_dirpath(path);
    cout << dir << " + " << filename << endl;
    
    return 0;
}