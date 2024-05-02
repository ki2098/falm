#include <string>
#include <cstdio>
#include <iostream>

using namespace std;

string wpath(string workdir, string str) {
    if (str[0] == '/') {
        return str;
    } else if (workdir.back() == '/') {
        return workdir + str;
    } else {
        return workdir + "/" + str;
    }
}

int main(int argc, char **argv) {
    string setup_file_path(argv[1]);
    int cutat = setup_file_path.find_last_of('/');
    string workdir, setupFile;
    if (cutat == string::npos) {
        workdir = ".";
        setupFile = setup_file_path;
    } else if (cutat == 0) {
        workdir = "/";
        setupFile = setup_file_path.substr(1);
    } else {
        workdir = setup_file_path.substr(0, cutat);
        setupFile = setup_file_path.substr(cutat + 1);
    }
    cout << workdir << " " << setupFile << endl;
    cout << wpath(workdir, setupFile) <<endl;

    string outputfile(argv[2]);
    cout << wpath(workdir, outputfile) << endl;
    
    return 0;
}