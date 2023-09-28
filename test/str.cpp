#include <string>
#include <stdio.h>

using namespace std;

class StrC {
public:
    string str;
    StrC(const string &_str = "") : str(_str) {}
};

int main() {
    char cstr[] = "12345";
    const char* cc = cstr;
    StrC str;
    printf("%s %s\n", str.str.c_str(), cc);
    cstr[0] = '0';
    printf("%s %s\n", str.str.c_str(), cc);
    str.str = cc;
    printf("%s %s\n", str.str.c_str(), cc);
    str.str = "ccccc";
    printf("%s %s\n", str.str.c_str(), cc);
    str.str = cc;
    printf("%s %s\n", str.str.c_str(), cc);
    str.str[0] = 'q';
    printf("%s %s\n", str.str.c_str(), cc);
    str.str = 77;
    printf("%s %s\n", str.str.c_str(), cc);
    return 0;
}