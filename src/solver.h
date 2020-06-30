#pragma once
#include "Tools.h"
#include "PBP.h"
#include "INIReader.h"
#include <string.h>


class solver_parameters;

class solver
{

protected:
    RandomNumberGenerator rng;
    PERMU::PBP* problem;
    stopwatch* timer;

public:

    solver();
    virtual ~solver();
    solver_parameters* params;
    virtual double solve() = 0;
    double average_k_runs(int k);
    void seed();
    void read_conf_file(string filename);
    void read_problem(string problemType, string filename);
    void seed(int seed);
};

class solver_parameters
{
    INIReader reader;
    string param_filename;
    string solver_name;

    public:
        solver_parameters(string filename);
        ~solver_parameters();
        int get_int(string param_name);
        double get_double(string param_name);
        string get_string(string param_name);
        bool get_bool(string param_name);
    
    private:
        void print_solver_parameter_read_error(string param_name);

};