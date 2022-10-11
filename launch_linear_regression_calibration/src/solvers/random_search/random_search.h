#pragma once
#include "solver.h"
#include <string.h>
class random_search : public solver
{


public:
    random_search(){};
    ~random_search(){};
    double solve();

private:
    double max_solver_time;
    string ls_operator_name;

};

