#include "random_search.h"
#include "Individual.h"
#include "permuevaluator.h"

double random_search::solve()
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



    max_solver_time = this->params->get_double("MAX_SOLVER_TIME");
    read_problem(this->params->get_string("PROBLEM_TYPE"), this->params->get_string("PROBLEM_PATH"));

    
    PERMU::CIndividual indiv(problem->GetProblemSize(), &this->rng);

    int random_permu[problem->GetProblemSize()];
    GenerateRandomPermutation(random_permu, problem->GetProblemSize(), &rng);

    indiv.reset(&rng);
    problem->Evaluate(&indiv);
    


    



    


    timer->tic();
    double f_best = indiv.f_value;



    while (timer->toc() < params->max_time && (problem->n_evals < params->max_evals || params->max_evals < 0))
    {
        GenerateRandomPermutation(indiv.genome, problem->GetProblemSize(), &rng);
        problem->Evaluate(&indiv);
        if(indiv.f_value > f_best)
        {
            f_best = indiv.f_value;
        }
    }

    //cout << f_best << endl;
    return f_best;
}
