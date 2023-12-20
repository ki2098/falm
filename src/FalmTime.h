#ifndef FALM_FALMTIME_H
#define FALM_FALMTIME_H

namespace Falm {

class FalmTime {
public:
    double start_time;
    double end_time;
    double delta_time;
    double timeavg_start_time;
    double timeavg_end_time;
    double output_start_time;
    double output_end_time;
    double output_interval;
};

}

#endif