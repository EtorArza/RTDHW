#include <string.h>
#include "Tools.h"
#include "solver.h"
#include "PBP.h"
#include "FitnessFunction_permu.h"

solver::solver()
{
    params = nullptr;
    problem = nullptr;
    timer = nullptr;
    timer = new stopwatch();
};
solver::~solver()
{
    if (problem != nullptr)
    {
        delete problem;
    }
    delete params;
    delete timer;
};

double solver::average_k_runs(int k)
{
    cout << "Not easy to parallelize. Better implement from outside class." << endl;
    exit(1);
}

void solver::seed()
{
    this->rng.seed();
}

void solver::seed(int seed)
{
    this->rng.seed(seed);
}

void solver::read_conf_file(string filename)
{
    if (params != nullptr)
    {
        delete params;
        params = nullptr;
    }
    params = new solver_parameters(filename);
}

void solver::read_problem(std::string problemType, std::string filename)
{
    PERMU::GetProblemInfo(problemType, filename, &this->problem);
}

solver_parameters::solver_parameters(string filename)
{
    param_filename = filename;
    reader = INIReader(filename);
    solver_name = get_string("SOLVER_NAME");
}

solver_parameters::~solver_parameters(){}

void solver_parameters::print_solver_parameter_read_error(string param_name)
{
    cerr << "Parameter -> "
         << param_name
         << " could not be read correctly from config file -> "
         << param_filename
         << " " << endl;
    exit(1);
}

int solver_parameters::get_int(string param_name)
{
    int res = reader.GetInteger("Global", param_name, -99999999);
    if (res == -99999999)
    {
        print_solver_parameter_read_error(param_name);
    }
    return res;
}

double solver_parameters::get_double(string param_name)
{
    double res = reader.GetReal("Global", param_name, -9999999999.9);
    if (res == -9999999999.9)
    {
        print_solver_parameter_read_error(param_name);
    }
    return res;
}

string solver_parameters::get_string(string param_name)
{
    string res = reader.Get("Global", param_name, "UNKNOWN");
    if (res == "UNKNOWN")
    {
        print_solver_parameter_read_error(param_name);
    }
    return res;
}

bool solver_parameters::get_bool(string param_name)
{
    bool res_1 = reader.GetBoolean("Global", param_name, true);
    bool res_2 = reader.GetBoolean("Global", param_name, false);

    if (res_1 != res_2)
    {
        print_solver_parameter_read_error(param_name);
    }
    return res_1;
}
