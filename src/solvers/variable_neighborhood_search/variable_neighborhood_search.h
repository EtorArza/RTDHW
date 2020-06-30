#pragma once
#include "solver.h"
#include <string.h>
class variable_neighborhood_search : public solver
{


public:
    variable_neighborhood_search(){};
    ~variable_neighborhood_search(){};
    double solve();

private:
    double max_solver_time;
    string ls_operator_name;

};

