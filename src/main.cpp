#include "solver.h"
#include "iterated_local_search.h"
#include "random_search.h"
#include "Tools.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "execute only with two parameters: the conf file path to use and the result file name." << std::endl;
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

stopwatch g_stopw = stopwatch();
g_stopw.tic();
int tot_evs = 0;
int max_evs_each =0;
double max_time_each =0;
for (int i = 0; i < N_EVALS; i++)
    {
        solver *sol;

        if (solver_name == "random_search")
        {
            sol = new random_search();
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
        max_evs_each = sol->params->max_evals;
        max_time_each = sol->params->max_time;
        res += (double)sol->solve();
        tot_evs += sol->problem->n_evals;
        delete sol;
    }
    res /= (double)N_EVALS;

    string cpu_name = system_exec("less /proc/cpuinfo | grep 'model name' | head -n 1 | awk -v FS=\"(:|)\" '{print $2}' | tr -d '\n'");

    string problem_type = system_exec(std::string() + "cat " + argv[1] + " | grep \"PROBLEM_TYPE\" |  awk -v FS=\"(=|;)\" '{print $2}' | tr -d ' \n'");
    string problem_path = system_exec(std::string() + "cat " + argv[1] + " | grep \"PROBLEM_PATH\" |  awk -v FS=\"(=|;)\" '{print $2}' | tr -d ' \n'");
    string solver_name_str = system_exec(std::string() + "cat " + argv[1] + " | grep \"SOLVER_NAME\" |  awk -v FS=\"(=|;)\" '{print $2}' | tr -d ' \n'");
    string operator_name = system_exec(std::string() + "cat " + argv[1] + " | grep \"OPERATOR\" |  awk -v FS=\"(=|;)\" '{print $2}' | tr -d ' \n'");
     

    string res_str = 
    cpu_name + "|" + 
    problem_type + "|" + 
    problem_path + "|" + 
    solver_name_str + "|" + 
    operator_name + "|" + 
    to_string(N_EVALS) + "|" +
    to_string(max_time_each) + "|" + 
    to_string(max_evs_each) + "|" + 
    to_string(g_stopw.toc()) + "|" + 
    to_string(tot_evs) + "|"+
    to_string(res) + "\n";
    append_line_to_file(argv[2], res_str);

    return 0;
}
