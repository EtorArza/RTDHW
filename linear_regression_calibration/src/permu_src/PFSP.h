/*
 *  PFSP.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#pragma once


#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "PBP.h"
using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::string;


class CIndividual;
namespace PERMU{

class PFSP : public PBP
{
	
public:
	
	/*
	 * The number of jobs of the problem.
	 */
	int m_jobs;
	
	/*
	 * The number of machines of the problem.
	 */
	int m_machines;
	
	/*
	 * The processing times matrix.
	 */
	int **m_processingtimes;
    
    /*
     * The time table for the processing times.
     */
    int * m_timeTable;


	// The constructor. It initializes a flowshop scheduling problem from a file.
	PFSP();
	
    // The destructor.
    virtual ~PFSP();
	
	/*
	 * Reads and PFSP instance with the given filename.
	 */
	int Read(string filename);
	

	double _Evaluate(int * permu);

	double _fitness_delta_swap(CIndividual *indiv, int i, int j){
		return fitness_delta_interchange(indiv, i, j);
	};

	double _fitness_delta_interchange(CIndividual *indiv, int i, int j){
		Swap(indiv->genome, i, j);
		double res = _Evaluate(indiv->genome);
		res -= indiv->f_value;
		Swap(indiv->genome, i, j);
		return res;
	};


	double _fitness_delta_insert(CIndividual *indiv, int i, int j){
		InsertAt(indiv->genome, i, j, m_jobs);
		double res = _Evaluate(indiv->genome);
		res -= indiv->f_value;
		InsertAt(indiv->genome, j, i, m_jobs);
		return res;
	};

    /*
     * Returns the size of the problem.
     */
    int GetProblemSize();

private: 
	
	
	
};


}

