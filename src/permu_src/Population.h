//
//  Population.h
//  RankingEDAsCEC
//
//  Created by Josu Ceberio Uribe on 11/19/13.
//  Copyright (c) 2013 Josu Ceberio Uribe. All rights reserved.
//

#pragma once


#include <string.h>
#include "Individual.h"
#include "PERMU_params.h"
#include "constants.h"

class RandomNumberGenerator;


namespace PERMU{

class PBP;
class PermuTools;
class Tabu;




class CPopulation
{


public:
  // if no rng object is provided, a new one is generated
  CPopulation(PBP *problem, PERMU::params* parameters);
  CPopulation(PBP *problem, RandomNumberGenerator* rng, PERMU::params* parameters);
  void init_class(PBP *problem, RandomNumberGenerator* rng, PERMU::params* parameters);


  virtual ~CPopulation();

  struct Better
  {
    bool operator()(CIndividual *a, CIndividual *b) const
    {
      return a->f_value > b->f_value;
    }
  } Better;

  // Vector of individuals that constitute the population.
  vector<CIndividual *> m_individuals;
  int n;
  int popsize;

 	double f_best;
	int* genome_best;

  void Print();
  void end_iteration(); // sort the population, check if the best solution was improved, and coompute neat imputs.
  void Reset();
  void change_params(PERMU::params* new_parameters);



  double* get_neat_input_individual_i(int i);
  void apply_neat_output_to_individual_i(double* output_neat, int i);
  void copy_individual_i_into_indiv_j(int i, int j);

  bool terminated;

   /*
   * Population info. Information about each individual. 
   * popinfo[i][j] has information about propertie i from individual j.
   */
  double **pop_info;
  stopwatch *timer;
  RandomNumberGenerator *rng;

private:

  double MAX_SOLVER_TIME;
  // evaluate the whole population. Only used to initialize the population.
  void evaluate_population();

  void SortPopulation(void);
  // Fill pop_info. Assumes the fitness values are computed, and that population is sorted.
  void get_population_info();


  void comp_relative_position();
  void comp_relative_time();
  void comp_distance();
  void comp_sparsity(bool first_time);
  void comp_r_number();
  void load_local_opt();
  void comp_order_sparsity(bool first_time);
  
  // Individual i is duplicated only if there is enough space. Additionally, the popsize is updated consecuently.
  void duplicate_individual_i(int i);
  
  // Individual only removed if there is enough space.
  void remove_individual_i(int i);

  void random_reinitialize_individual_i(int i);

  
  PBP * problem;
  PermuTools *pt;
  double *templ_double_array;
  double *templ_double_array2;
  double relative_time();

  double iteration_geom;
  double iteration_geom_coef;

  vector<int> indexes_to_be_removed;
  vector<int> indexes_to_be_duplicated;


  /* 
  * In this case, 0 means highly cramped, 1 means highly sparse.
  * Each permutation is compared with the next and previous permutations.
  * For example, if permutations on position 5 and 6 are the same, then result_vector[5] = result_vector[6] = 0
  */
  void get_info_Hamming_distance_from_adjacent_permus(double *result_vector);
  int **permus; //this is only intended to temporally store the permutations for computations.

  // copy the reference of the permutations in the individuals to a matrix of permutations for analisys in sparsity and distances.
  void copy_references_of_genomes_from_individuals_to_permus();

  // move permutation based on coefs. and other permus.
  // first choose a permu proportionally to its weight. 
  // Then if weight is positive move towards, otherwise, move away from.
  // to permuevaluator.h contains which permutations are considered.
  void move_individual_i_based_on_coefs(double* coef_list, int i, PERMU::operator_t operator_id, double accept_or_reject_worse);


 
};

}