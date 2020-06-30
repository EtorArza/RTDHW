#include "solver.h"
#include "iterated_local_search.h"
#include "variable_neighborhood_search.h"
#include "Tools.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "execute only with one parameter: the conf file path to use." << std::endl;
        exit(1);
    }
    solver *param_sol;
    param_sol = new iterated_local_search();
    param_sol->read_conf_file(argv[1]);

    int N_EVALS = param_sol->params->get_int("N_EVALS");
    int N_OF_THREADS = param_sol->params->get_int("N_OF_THREADS");

    std::string solver_name = param_sol->params->get_string("SOLVER_NAME");

    delete param_sol;



    double res = 0.0;

#pragma omp parallel for num_threads(N_OF_THREADS)
for (int i = 0; i < N_EVALS; i++)
    {
        solver *sol;

        if (solver_name == "variable_neighborhood_search")
        {
            sol = new variable_neighborhood_search();
        }
        else if (solver_name == "iterated_local_search")
        {
            sol = new iterated_local_search();
        }
        else
        {
            std::cout << "ERROR, solver type '" << solver_name << "' not recognized." << endl;
            exit(1);
        }
        

        sol->read_conf_file(argv[1]);
        res += (double)sol->solve();
        delete sol;
    }
    res /= (double)N_EVALS;

    string res_str = to_string(res);
    append_line_to_file("result.txt", res_str);

    return 0;
}
