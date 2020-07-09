/*
 *  LOP.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 11/21/11.
 *  Copyright 2011 University of the Basque Country. All rights reserved.
 *
 */

#pragma once

#include "Tools.h"
#include "PBP.h"
#include "Individual.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
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


class LOP : public PBP
{
	
public:
	
    /*
     * Entries matrix of the LOP.
     */
	double ** m_matrix;
    
	/*
	 * The size of the problem.
	 */
	int n;
		

    /*
     * The constructor.
     */
	LOP();
	

    virtual ~LOP();
	int Read(string filename);
    int GetProblemSize();




	double _fitness_delta_swap(CIndividual *indiv, int i, int j);
	double _fitness_delta_interchange(CIndividual *indiv, int i, int j);
	double _fitness_delta_insert(CIndividual *indiv, int i, int j);



private:
    double _Evaluate(int *permu);
};





}