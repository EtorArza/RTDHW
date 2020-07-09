#include "iterated_local_search.h"
#include "solver.h"
#include "PBP.h"
#include "Tabu.h"
#include "Individual.h"
#include "permuevaluator.h"

double iterated_local_search::solve()
{
    ls_operator_name = this->params->get_string("OPERATOR");

    PERMU::operator_t ls_operator;

    if (ls_operator_name == "insert")
    {
        ls_operator = PERMU::INSERT;
    }
    else if (ls_operator_name == "exchange")
    {
        ls_operator = PERMU::EXCH;
    }
    else if (ls_operator_name == "swap")
    {
        ls_operator = PERMU::SWAP;
    }

    read_problem(this->params->get_string("PROBLEM_TYPE"), this->params->get_string("PROBLEM_PATH"));

    
    PERMU::CIndividual indiv(problem->GetProblemSize(), &this->rng);

    int random_permu[problem->GetProblemSize()];
    GenerateRandomPermutation(random_permu, problem->GetProblemSize(), &rng);

    indiv.reset(&rng);
    problem->Evaluate(&indiv);
    
    problem->tab = new PERMU::Tabu(&rng, problem->GetProblemSize());
    problem->tab->tabu_coef_neat = 0.0;

    if (this->params->get_bool("USE_TABU"))
    {
        while (problem->tab->tabu_length < this->params->get_int("TABU_LENGTH"))
        {
            problem->tab->increase_tabu_size();
        }
        problem->tab->tabu_coef_neat = 1.0;
    }
    



    


    timer->tic();
    double f_best = indiv.f_value;

    #define RESET_METHOD_RANDOM_REINITIALIZE 1
    #define RESET_METHOD_MOVE_TOWARDS_RANDOM 2

    int RESET_METHOD;
    int STEPS_TO_MOVE_TOWARDS_RANDOM;
    if (this->params->get_string("RESET_METHOD") == "reinitialize")
    {
        RESET_METHOD = RESET_METHOD_RANDOM_REINITIALIZE;
    }
    else if(this->params->get_string("RESET_METHOD") == "move_towards_random")
    {
        RESET_METHOD = RESET_METHOD_MOVE_TOWARDS_RANDOM;
        STEPS_TO_MOVE_TOWARDS_RANDOM = this->params->get_int("STEPS_TO_MOVE_TOWARDS_RANDOM");
    }else
    {
        cout << "Error, reset method not recognized." << endl;
        exit(1);
    }
    

    while (timer->toc() < params->max_time && problem->n_evals < params->max_evals)
    {
        if (indiv.is_local_optimum[ls_operator])
        {
            if (RESET_METHOD == RESET_METHOD_RANDOM_REINITIALIZE)
            {
                GenerateRandomPermutation(indiv.genome, problem->GetProblemSize(), &rng);
                problem->Evaluate(&indiv);
            }
            else if (RESET_METHOD == RESET_METHOD_MOVE_TOWARDS_RANDOM)
            {
                GenerateRandomPermutation(random_permu, problem->GetProblemSize(), &rng);
                for (int i = 0; i < STEPS_TO_MOVE_TOWARDS_RANDOM; i++)
                {
                    problem->move_indiv_towards_reference(&indiv, random_permu, ls_operator);
                }
            }
            indiv.is_local_optimum[0] = false;
            indiv.is_local_optimum[1] = false;
            indiv.is_local_optimum[2] = false;

        }
        problem->local_search_iteration(&indiv, ls_operator);
        if(indiv.f_value > f_best)
        {
            f_best = indiv.f_value;
        }
    }

    //cout << f_best << endl;
    return f_best;
}
