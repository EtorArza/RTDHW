//
//  Individual.h
//  RankingEDAsCEC
//
//  Created by Josu Ceberio Uribe on 11/19/13.
//  Copyright (c) 2013 Josu Ceberio Uribe. All rights reserved.
//

#pragma once


#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <limits.h>
#include <iostream>
#include "Tools.h"
#include "Tabu.h"
#include "permuevaluator.h"


using namespace std;
using std::istream;
using std::ostream;

namespace NEAT{
class CpuNetwork;
}

namespace PERMU{

class CIndividual
{
public:
	// The constructor. The constructed individual has
	// all zeroes as its genes.
	CIndividual(int length, RandomNumberGenerator* rng);
	void reset(RandomNumberGenerator* rng);
	~CIndividual();
	void SetGenome(int *genes);
	void PrintGenome();
	CIndividual *Clone();

	/*
     * Output operator.
     */
	friend ostream &operator<<(ostream &os, CIndividual *&individual);

	/*
     * Input operator.
     */
	friend istream &operator>>(istream &is, CIndividual *&individual);

	int n;
	int *genome; //The genes of the individual, or the permutation.
	int id;		 // a unique identifier for each individual in the pop.
	double f_value;
	bool is_local_optimum[PERMU::N_OPERATORS] = {false, false, false}; // is_local_optimum[OPERATOR_ID] contains if it is optimum or not.
	double f_best;
	int* genome_best;
	Tabu* tab;


	// pop_info
	double relative_pos = 0;
	double relative_time = 0;
	double sparsity = 0;
	double order_sparsity = 0;
	double distance = 0;
	
	std::vector<double> activation;



private:
	static int n_indivs_created;
};

}

