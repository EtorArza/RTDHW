#pragma once
#include "solver.h"
#include <string.h>
class iterated_local_search : public solver
{


public:
    iterated_local_search(){};
    ~iterated_local_search(){};
    double solve();

private:
    string ls_operator_name;

};

