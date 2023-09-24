#include <iostream>
#include <vector>

class vvec {
    std::vector<double> vv[5];
    vvec() {
        
    }
};

template<typename T>
class vecvec{
public:
    std::vector<T> vv[5];
    std::vector<T> v;
    std::vector<int> vvi[5];

    vecvec() {
        
    }
};