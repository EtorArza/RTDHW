/*
 *  QAP.h
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
#include "Individual.h"

using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::stringstream;
using std::string;

namespace PERMU{


class QAP : public PBP
{
	
public:
	
    /*
     * The matrix of distances between the cities.
     */
	int ** m_distance_matrix;
	
    /*
     * The flow matrix.
     */
	int ** m_flow_matrix;
	
	/*
	 * The number of jobs of the problem.
	 */
	int n;
	
	/*
     * The constructor. It initializes a QAP from a file.
     */
	QAP();
	
    /*
     * The destructor.
     */
    virtual ~QAP();
	
	int Read(string filename);
    int GetProblemSize();



	double fitness_delta_swap(CIndividual *indiv, int i, int j);
	double fitness_delta_interchange(CIndividual *indiv, int i, int j);
	double fitness_delta_insert(CIndividual *indiv, int i, int j);

protected:	
    double _Evaluate(int *permu);


};

}