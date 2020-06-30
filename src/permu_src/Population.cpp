//
//  Population.cc
//  RankingEDAsCEC
//
//  Created by Josu Ceberio Uribe on 11/19/13.
//  Copyright (c) 2013 Josu Ceberio Uribe. All rights reserved.
//

#include "Population.h"
#include "PBP.h"
#include "Parameters.h"
#include "Tools.h"
#include "Tabu.h"
#include "permuevaluator.h"
#include <assert.h>
#include <float.h>
#include "PERMU_params.h"

using std::cerr;
using std::cout;
using std::endl;
using std::istream;
using std::ostream;



namespace PERMU{

void CPopulation::init_class(PBP *problem, RandomNumberGenerator* rng, PERMU::params* parameters){
    this->rng = rng;
    this->problem = problem;
    this->popsize = MIN_POPSIZE;
    this->MAX_SOLVER_TIME = parameters-> MAX_SOLVER_TIME;
    this->n = problem->GetProblemSize();
    genome_best = new int[n];
    f_best = -DBL_MAX;

    indexes_to_be_removed.reserve(MAX_POPSIZE);
    indexes_to_be_duplicated.reserve(MAX_POPSIZE);

    GenerateRandomPermutation(this->genome_best, n, this->rng);
    templ_double_array = new double[MAX_POPSIZE];
    templ_double_array2 = new double[MAX_POPSIZE];

    for (int i = 0; i < MAX_POPSIZE; i++)
    {
        indexes_to_be_removed[i] = -1;
        indexes_to_be_duplicated[i] = -1;
    }

    m_individuals.resize(MAX_POPSIZE);

    pop_info = new double *[MAX_POPSIZE];
    permus = new int *[MAX_POPSIZE]; // this contains the references to te permus in the individuals, so no initialization/destruction.

    //Initialize population with random solutions
    for (int i = 0; i < MAX_POPSIZE; i++)
    {
        m_individuals[i] = new CIndividual(n, this->rng);
    }

    for (int i = 0; i < MAX_POPSIZE; i++)
    {
        pop_info[i] = new double[PERMU::__sensor_N];
    }
    pt = new PermuTools(n, rng);
    timer = new stopwatch();
    terminated = false;
    evaluate_population();
    comp_sparsity(true);
    comp_order_sparsity(true);
    end_iteration();
}

CPopulation::CPopulation(PBP *problem, PERMU::params* parameters)
{
    RandomNumberGenerator* tmp_rng = new RandomNumberGenerator();
    init_class(problem, tmp_rng, parameters);
}


CPopulation::CPopulation(PBP *problem, RandomNumberGenerator* rng, PERMU::params* parameters)
{
    init_class(problem, rng, parameters);
}

void CPopulation::Reset(){
    f_best = -DBL_MAX;
    GenerateRandomPermutation(this->genome_best, n, rng);
    for (int i = 0; i < MAX_POPSIZE; i++)
    {   
        auto tmp = std::vector<double>();
        std::swap(tmp, m_individuals[i]->activation);
        m_individuals[i]->reset(rng);
        std::swap(tmp, m_individuals[i]->activation);
    }
    this->popsize = MIN_POPSIZE;
    terminated = false;
    timer->tic();
    evaluate_population();
    comp_sparsity(true);
    comp_order_sparsity(true);
    end_iteration();
}

void CPopulation::copy_individual_i_into_indiv_j(int i, int j)
{
    m_individuals[j]->reset(rng);
    memcpy(m_individuals[j]->is_local_optimum, m_individuals[i]->is_local_optimum, sizeof(m_individuals[i]->is_local_optimum));
    memcpy(m_individuals[j]->genome, m_individuals[i]->genome, sizeof(int) * m_individuals[i]->n);
    memcpy(m_individuals[j]->genome_best, m_individuals[i]->genome_best, sizeof(int) * m_individuals[i]->n);
    m_individuals[j]->f_best =  m_individuals[i]->f_best;
    m_individuals[j]->f_value =  m_individuals[i]->f_value;
}

void CPopulation::duplicate_individual_i(int i)
{
    if (popsize < MAX_POPSIZE)
    {
        copy_individual_i_into_indiv_j(i,popsize);
        popsize++;
    }
}

void CPopulation::remove_individual_i(int i)
{
    if (popsize < MIN_POPSIZE)
    {
        for (int k = i; k < popsize; k++)
        {
            swap(m_individuals[k], m_individuals[i]);
        }
        popsize--;
    }
}

void CPopulation::random_reinitialize_individual_i(int i)
{
    m_individuals[i]->reset(rng);
    problem->Evaluate(m_individuals[i]);
}

/*
 * Destructor function.
 */
CPopulation::~CPopulation()
{
    for (int i = 0; i < MAX_POPSIZE; i++)
    {
        delete[] pop_info[i];
    }
    for (int i = 0; i < MAX_POPSIZE; i++)
    {
        delete m_individuals[i];
    }
    
    m_individuals.clear();
    delete timer;
    timer=NULL;
    delete pt;
    pt=NULL;
    delete rng;
    rng=NULL;    
    delete[] pop_info;
    pop_info=NULL;
    delete[] permus;
    permus=NULL;
    delete[] genome_best;
    genome_best=NULL;
    delete[] templ_double_array;
    templ_double_array=NULL;
    delete[] templ_double_array2;
    templ_double_array2=NULL;
}



void CPopulation::end_iteration(){
    for (int i = 0; i < indexes_to_be_duplicated.size(); i++)
    {
        duplicate_individual_i(indexes_to_be_duplicated[i]);
    }
    indexes_to_be_duplicated.clear();
    
    for (int i = indexes_to_be_duplicated.size() -1 ; i >= 0; i--)
    {
        remove_individual_i(indexes_to_be_removed[i]);
    }
    indexes_to_be_removed.clear();

    SortPopulation();
    get_population_info();
    if(timer->toc() > this->MAX_SOLVER_TIME)
    {
        terminated = true;
    }
    //PrintMatrix(pop_info, this->popsize, NEAT::__sensor_N);
}


/*
 * Prints the current population.
 */
void CPopulation::Print()
{   cout << "---" << endl;
    for (int i = 0; i < popsize; i++)
        PrintArray(pop_info[i], PERMU::__sensor_N);
    cout << "---" << endl;
}

double *CPopulation::get_neat_input_individual_i(int i)
{
    return pop_info[i];
}

// Apply modifications to solution i stored in m_individuals[i] based on the oputput from the controller.
void CPopulation::apply_neat_output_to_individual_i(double* output_neat, int i){

    if (output_neat[PERMU::REMOVE_OR_CLONE] > CUTOFF_0)
    {
        indexes_to_be_duplicated.push_back(i);
    }
    else if (output_neat[PERMU::REMOVE_OR_CLONE] < - CUTOFF_0)
    {
        indexes_to_be_removed.push_back(i);
        return;
    }

    if (output_neat[PERMU::RANDOM_REINITIALIZE] > CUTOFF_0)
    {
        random_reinitialize_individual_i(i);
        return;
    }


    double accept_or_reject_worse = output_neat[PERMU::accept_or_reject_worse];
    m_individuals[i]->tab->tabu_coef_neat = output_neat[(int) PERMU::TABU];

    if
    (
        (-CUTOFF_0 < output_neat[PERMU::ls_nothing_move] && output_neat[PERMU::ls_nothing_move] < CUTOFF_0) ||
        (are_doubles_equal(sum_abs_val_slice_vec(output_neat, 1, 1+PERMU::N_OPERATORS), 0.0))
    )
    {return;}


    else if(output_neat[0] < -CUTOFF_0){ // Local-search iteration.
        //#TODO check if unconnected output is 0.
        PERMU::operator_t operator_id = (PERMU::operator_t) argmax(output_neat + 1, PERMU::N_OPERATORS);
        this->problem->local_search_iteration(m_individuals[i], operator_id);
    }else if(output_neat[0] > CUTOFF_0){ // Move-with coeficients.
        PERMU::operator_t operator_id = (PERMU::operator_t) argmax(output_neat + 1, PERMU::N_OPERATORS);
        double* coef = output_neat + (PERMU::__output_N - PERMU::N_PERMU_REFS);
        this->move_individual_i_based_on_coefs(coef, i, operator_id, accept_or_reject_worse);
        assert(isPermutation(this->m_individuals[i]->genome, this->n));

    }

    if (output_neat[PERMU::CHANGE_TABU_SIZE] > CUTOFF_0)
    {
        m_individuals[i]->tab->increase_tabu_size();
    }
    else if (output_neat[PERMU::CHANGE_TABU_SIZE] < -CUTOFF_0)
    {
        m_individuals[i]->tab->decrease_tabu_size();
    }

}




/*
 * Sorts the individuals in the population in decreasing order of the fitness value.
 */
void CPopulation::SortPopulation()
{
    sort(m_individuals.begin(), m_individuals.end(), Better);
    if (m_individuals[0]->f_value > this->f_best)
    {
        this->f_best = m_individuals[0]->f_value;
        copy_array(this->genome_best, m_individuals[0]->genome, n);
    }
    
}

// Evaluate the whole population
void CPopulation::evaluate_population()
{
    for (int i = 0; i < popsize; i++)
    {
        problem->Evaluate(m_individuals[i]);
    }
}


void CPopulation::get_population_info(){
    comp_relative_position();
    comp_relative_time();
    comp_distance();
    comp_sparsity(false);
    comp_order_sparsity(false);
    load_local_opt();
    for (int i = 0; i < popsize; i++)
    {
        pop_info[i][PERMU::RELATIVE_POPSIZE] = (double) (this->popsize - MIN_POPSIZE) / (double) MAX_POPSIZE ;
        pop_info[i][PERMU::RELATIVE_TABU_SIZE] = m_individuals[i]->tab->return_current_relative_tabu_size();
    }
}


void CPopulation::comp_relative_position()
{
    for (int i = 0; i < this->popsize; i++)
    {
        double res =  (double)i / (double)this->popsize;
        this->m_individuals[i]->relative_pos = res;
        pop_info[i][PERMU::RELATIVE_POSITION] = res;
    }
}

void CPopulation::comp_relative_time()
{
    for (int i = 0; i < this->popsize; i++)
    {
        double res = timer->toc() / this->MAX_SOLVER_TIME;
        this->m_individuals[i]->relative_time = res;
        pop_info[i][PERMU::RELATIVE_TIME] = res;
    }
    return ;
}

void CPopulation::comp_distance()
{

    // use the ranking of the differences in fitness with respect to the previous one. 
    templ_double_array[0] = DBL_MAX;
    for (int i = 1; i < this->popsize; i++)
    {   
        double val = m_individuals[i-1]->f_value -  m_individuals[i]->f_value;
        templ_double_array[i] = val;
    }

    compute_order_from_double_to_double(templ_double_array, this->popsize, templ_double_array2);


    // copy normalized values into individuals
    for (int i = 0; i < this->popsize; i++)
    {   
        double val = templ_double_array2[i] / (double) this->popsize;
        m_individuals[i]->distance = val;
        pop_info[i][PERMU::DISTANCE] = val;
    }

//region old_implementation_hamming_distance
    // // minimum of Hamming distance between the previous one and the next one 
    // // First, compute the distance of each permu with respect the  next permu
    // pop_info[0][NEAT::DISTANCE] = (double)Hamming_distance(m_individuals[0]->genome, m_individuals[1]->genome, n);
    // for (int i = 1; i < this->popsize - 1; i++)
    // {
    //     pop_info[i][NEAT::DISTANCE] = (double)Hamming_distance(m_individuals[i]->genome, m_individuals[i + 1]->genome, n);
    // }
    // pop_info[this->popsize - 1][NEAT::DISTANCE] = pop_info[this->popsize - 2][NEAT::DISTANCE];

    // // Then, assign to result_vector[i], the minimun of the distances between the next an the prev permus.
    // double distance_respect_to_previous = pop_info[0][NEAT::DISTANCE];
    // double temp;
    // for (int i = 1; i < this->popsize - 1; i++)
    // {
    //     temp = pop_info[i][NEAT::DISTANCE];
    //     pop_info[i][NEAT::DISTANCE] = MIN(pop_info[i][NEAT::DISTANCE], distance_respect_to_previous);
    //     distance_respect_to_previous = temp;
    // }

    // // Finally, normalize the values for them to be between 0 and 1.
    // for (int i = 0; i < this->popsize; i++)
    // {
    //     pop_info[i][NEAT::DISTANCE] /= (double)n;
    // }

    // // copy values into individuals
    // for (int i = 0; i < this->popsize; i++)
    // {
    //     m_individuals[i]->distance = pop_info[i][NEAT::DISTANCE];
    // }
//endregion old_implementation_hamming_distance

}


void CPopulation::comp_sparsity(bool first_time){

    copy_references_of_genomes_from_individuals_to_permus();


    const int COMPUTE_CONSENSUS_EVERY = 20;
    static int iterations_left_for_next_computation_of_consensus = 0;
    
    if (first_time)
    {
        iterations_left_for_next_computation_of_consensus = 0;
    }

    if(iterations_left_for_next_computation_of_consensus == 0)
    {
        pt->compute_hamming_consensus(this->permus, this->popsize);
        iterations_left_for_next_computation_of_consensus = COMPUTE_CONSENSUS_EVERY;
    }

    iterations_left_for_next_computation_of_consensus--;

    for (int i = 0; i < this->popsize; i++)
    {
        m_individuals[i]->sparsity = 1.0 - pt->compute_normalized_hamming_distance_to_consensus(permus[i]);
        pop_info[i][PERMU::SPARSITY] = m_individuals[i]->sparsity;
    }
}


void CPopulation::comp_order_sparsity(bool first_time){


    const int COMPUTE_CONSENSUS_EVERY = 20;
    static int iterations_left_for_next_computation_of_consensus = 0;

    if (first_time)
    {
        iterations_left_for_next_computation_of_consensus = 0;
    }
    

    if(iterations_left_for_next_computation_of_consensus == 0)
    {
        pt->compute_kendall_consensus_borda(this->permus, popsize);
        iterations_left_for_next_computation_of_consensus = COMPUTE_CONSENSUS_EVERY;
    }

    iterations_left_for_next_computation_of_consensus--;


    for (int i = 0; i < this->popsize; i++)
    {
        m_individuals[i]->order_sparsity = 1.0 - pt->compute_normalized_kendall_distance_to_consensus_fast_approx(permus[i]);
        pop_info[i][PERMU::ORDER_SPARSITY] = m_individuals[i]->order_sparsity;
    }
}


// void CPopulation::comp_r_number()
// {
//     for (int i = 0; i < this->popsize; i++)
//     {
//         double res =  rng->random_0_1_double();
//         pop_info[i][NEAT::R_NUMBER] = res;
//     }
// }

void CPopulation::load_local_opt(){
for (int i = 0; i < this->popsize; i++)
{
    pop_info[i][PERMU::OPT_SWAP] = (double) m_individuals[i]->is_local_optimum[PERMU::SWAP];
    pop_info[i][PERMU::OPT_EXCH] = (double) m_individuals[i]->is_local_optimum[PERMU::EXCH];
    pop_info[i][PERMU::OPT_INSERT] = (double) m_individuals[i]->is_local_optimum[PERMU::INSERT];
}
}

// void CPopulation::take_action_with_action_id(int permutation_index, int action_id, PBP *problem)
// {
//     switch (action_id)
//     {
//     case ACTION_1_LOCAL_SEARCH_ITERATION:
//         problem->local_search_iteration_insertion(m_individuals[permutation_index]);
//         break;
//     case ACTION_2_MOVE_AWAY_FROM_WORSE:
//         if (permutation_index != this->popsize - 1)
//         {
//             problem->move_permutation_away_reference_with_insertion(m_individuals[permutation_index], m_individuals[permutation_index + 1]);
//         }
//         break;
//     case ACTION_3_MOVE_TOWARDS_BEST:
//         if (permutation_index != 0)
//         {
//             problem->move_permutation_towards_reference_with_insertion(m_individuals[permutation_index], m_individuals[0]);
//         }
//         break;
//     case ACTION_4_MOVE_TOWARDS_BETTER:
//         if (permutation_index != 0)
//         {
//             problem->move_permutation_towards_reference_with_insertion(m_individuals[permutation_index], m_individuals[permutation_index - 1]);
//         }
//         break;
//     default:
//         cout << "ERROR, action_id not recognized" << endl;
//         exit(1);
//     }
// }



void CPopulation::copy_references_of_genomes_from_individuals_to_permus(){
    for (int i = 0; i < this->popsize; i++)
    {
        permus[i] = m_individuals[i]->genome;
    }
}
 

void CPopulation::move_individual_i_based_on_coefs(double* coef_list, int i, PERMU::operator_t operator_id, double accept_or_reject_worse){

    int idx = pt->choose_permu_index_to_move(coef_list, rng);
    if (idx == -1){
        return;
    }


    assert(isPermutation(this->m_individuals[i]->genome, this->n)) ;

    bool towards = coef_list[idx] > 0;
    idx += (PERMU::__output_N - PERMU::N_PERMU_REFS);

    int* ref_permu;
    
    assert(isPermutation(this->m_individuals[i]->genome, this->n)) ;


    switch (idx)
    {
    case PERMU::c_hamming_consensus:
        ref_permu = pt->hamming_mm_consensus;
        break;
    case PERMU::c_kendall_consensus:
        ref_permu = pt->kendall_mm_consensus;
        break;
    case PERMU::c_pers_best:
        ref_permu = m_individuals[i]->genome_best;
        break;
    case PERMU::c_best_known:
        ref_permu = this->genome_best;
        break;
    case PERMU::c_above:
        if (i == 0)
        {
            ref_permu = this->m_individuals[1]->genome;
        }else
        {
            ref_permu = this->m_individuals[i-1]->genome;
        }
        break;
    
    default:
        cout << "error: a permutation must be chosen to move towards/away from.";
        exit(1);
        break;
    }


    if(towards){
        problem->move_indiv_towards_reference(m_individuals[i], ref_permu, operator_id, accept_or_reject_worse);
        assert(isPermutation(this->m_individuals[i]->genome, this->n)) ;

    }else
    {
        problem->move_indiv_away_reference(m_individuals[i], ref_permu, operator_id, accept_or_reject_worse);
        assert(isPermutation(ref_permu, this->n)) ;
        assert(isPermutation(this->m_individuals[i]->genome, this->n)) ;

    }
}

}