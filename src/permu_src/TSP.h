/*
 *  TSP.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#pragma once


#include "PBP.h"
#include "Tools.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::string;

namespace PERMU{

class TSP : public PBP
{
	
public:
	
    /*
     * Matrix of distances between the cities.
     */
	double ** m_distance_matrix;
	
	/*
	 * The number of cities.
	 */
	int m_size;
	
	/*
     * The constructor.
     */
	TSP();
	
    /*
     * The destructor.
     */
    virtual ~TSP();
	


	// The cpp inplementation is for the assymetric TSP, however, it is currently redirected to the Read2 function.
	int Read(string filename);
    

	// Read the cordenates of cities from file, thus symmertric TSP problems.
	int Read2(string filename);


	double _Evaluate(int * genes);

  	double _fitness_delta_swap(CIndividual *indiv, int i, int j);
	double _fitness_delta_interchange(CIndividual *indiv, int i, int j);
	double _fitness_delta_insert(CIndividual *indiv, int i, int j);
  
    /*
     * Returns the size of the problem.
     */
    int GetProblemSize();
private:
	
};
}
