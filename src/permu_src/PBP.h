
#pragma once

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "Individual.h"
#include "Tools.h"
#include  <mutex>

namespace PERMU{


using std::string;
using std::stringstream;
class Tabu;


class PBP
{

	static std::mutex mut;

  public:
	PBP();

	/*
	 * Virtual functions to be defined with each permutation problem.
	 */
	virtual ~PBP();
	virtual int Read(string filename) = 0;
	virtual int GetProblemSize() = 0;


	/*
	 * Functions that are valid for all permutation problems. 
	 * The use of this functions requires the fitness value of the individual to be previously computed.
	 * No need to define them in the problem class.
	 */
	void local_search_iteration(CIndividual *indiv, PERMU::operator_t operator_id);
	void move_indiv_towards_reference(CIndividual* indiv, int* ref_permu, PERMU::operator_t operator_id, double accept_or_reject_worse=1.0);
	void move_indiv_away_reference(CIndividual* indiv, int* ref_permu, PERMU::operator_t operator_id, double accept_or_reject_worse=1.0);
	void Evaluate(CIndividual *indiv); // update the f_value of the individuals.
	double Evaluate(int *genome); // update the f_value of the individuals.
	int Read_with_mutex(string filename);
	void load_rng(RandomNumberGenerator *rng);
	Tabu* tab;
	RandomNumberGenerator *rng;


  protected:

	void apply_operator_with_fitness_update(CIndividual *indiv, double delta, int i, int j, PERMU::operator_t operator_id, double accept_or_reject_worse=1.0);
	void apply_operator_with_fitness_update(CIndividual *indiv, int i, int j, PERMU::operator_t operator_id, double accept_or_reject_worse=1.0);


	// This function needs to be executed on problem read.
	int item_i_after_operator(int *permu, int idx, PERMU::operator_t operator_id, int i, int j);
	void initialize_variables_PBP(int problem_size);
	int *_random_permu1;
	int *_random_permu2;
	int *_random_permu3;
	int problem_size_PBP;
	string filename;



	/*
	* Functions to be defined in child class.
	*
	*/
	// The f_value of the individuals does not change in this functions. Just return the delta.
	virtual double _Evaluate(int * permu) = 0;
	virtual double fitness_delta_swap(CIndividual *indiv, int i, int j) = 0;
	virtual double fitness_delta_interchange(CIndividual *indiv, int i, int j) = 0;
	virtual double fitness_delta_insert(CIndividual *indiv, int i, int j) = 0;


  private:
	void obtain_indexes_step_towards(int *permu, int *ref_permu, int* i, int* j, PERMU::operator_t operator_id);
	void obtain_indexes_step_away(int *permu, int *ref_permu, int* i, int* j, PERMU::operator_t operator_id);
	bool rng_deleted = false;

};
}