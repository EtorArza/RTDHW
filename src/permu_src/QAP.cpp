/*
 *  QAP.cpp
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#include "QAP.h"
#include <assert.h>
#include "Tools.h"

namespace PERMU{


/*
 *Class constructor.
 */
QAP::QAP()
{
	
}

/*
 * Class destructor.
 */
QAP::~QAP()
{
	for (int i=0;i<n;i++)
	{
		delete [] m_distance_matrix[i];
		delete [] m_flow_matrix[i];
	}
	delete [] m_flow_matrix;
	delete [] m_distance_matrix;

}



int QAP::Read(string filename)
{
	char line[5096]; // variable for input value
	ifstream indata;
	indata.open(filename.c_str(),ios::in);
	int num=0;
	while (!indata.eof())
	{
		//LEER LA LINEA DEL FICHERO
		indata.getline(line, 5096);
		stringstream ss;
		string sline;
		ss << line;
		ss >> sline;
        //cout<<"line: "<<line<<endl;
		if (num==0)
		{
			//OBTENER EL TAMAÑO DEL PROBLEMA
			n = atoi(sline.c_str());
			m_distance_matrix = new int*[n];
			m_flow_matrix = new int*[n];
			for (int i=0;i<n;i++)
			{
				m_distance_matrix[i]= new int[n];
				m_flow_matrix[i] = new int[n];
			}
		}
		else if (1<=num && num<=n)
		{
			//LOAD DISTANCE MATRIX
			char * pch;
			pch = strtok (line," ");
			int distance=atoi(pch);
			m_distance_matrix[num-1][0]=distance;
			for (int i=1;i < n; i++)
			{
				pch = strtok (NULL, " ,.");
				distance=atoi(pch);
				m_distance_matrix[num-1][i]=distance;
			}
		}
		else if (num>n && num<=(2*n))
		{
			//LOAD FLOW MATRIX
			char * pch;
			pch = strtok (line," ");
			int weight=atoi(pch);
			m_flow_matrix[num-n-1][0]=weight;
			for (int i=1;i < n; i++)
			{
				pch = strtok (NULL, " ,.");
				weight=atoi(pch);
				m_flow_matrix[num-n-1][i]=weight;
			}
		}
 
		else
		{
			break;
		}/*
        //LOAD DISTANCE MATRIX
        else{
            int distance;
            if (row>=m_size){
                //flow_matrix
                char * pch;
                pch = strtok (line," ");
                while (pch != NULL)
                {
                    distance=atoi(pch);
                    m_flow_matrix[row-m_size][col]=distance;
                    //cout<<"flow: "<<distance<<" row: "<<row<<" col: "<<col<<endl;
                    col++;
                    if (col==m_size){
                        col=0;
                        row++;
                    }
                    pch = strtok (NULL, " ,.");
                }
            }
            else{
                //distance_matrix
                char * pch;
                pch = strtok (line," ");
                while (pch != NULL)
                {
                    distance=atoi(pch);
                 //   cout<<"dist: "<<distance<<" row: "<<row<<" col: "<<col<<endl;
                    m_distance_matrix[row][col]=distance;
                    col++;
                    if (col==m_size){
                        col=0;
                        row++;
                    }
                    pch = strtok (NULL, " ,.");
                }
            }
        }*/
        
		num++;
	}
    //PrintMatrix(m_distance_matrix, m_size, m_size, "");
    //PrintMatrix(m_flow_matrix, m_size, m_size, "");
    //exit(1);
	initialize_variables_PBP(n);


	return (n);
}

/*
 * This function evaluates the individuals for the QAP problem.
 */
double QAP::_Evaluate(int * genes)
{
	double fitness=0;
	int FactA, FactB;
	int distAB, flowAB, i ,j;
	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
		{
			FactA = genes[i];
			FactB = genes[j];
			
			distAB= m_distance_matrix[i][j];
			flowAB= m_flow_matrix[FactA][FactB];

            
			fitness += distAB*flowAB;			
		}
	}
	
	return -fitness;
}

	double QAP::fitness_delta_swap(CIndividual *indiv, int i, int j){
		assert(j == i+1);
		return fitness_delta_interchange(indiv, i, j);
	}

	double QAP::fitness_delta_interchange(CIndividual *indiv, int i, int j){
		int new_fitness_delta = 0;
		int el_at_pos_i_in_sigma_2;
		if (i > j)
		{
			int aux = i;
			i = j;
			j = aux;
		}else if(i==j){
			return 0;
		}

        for (int k = 0; k < i; k++)
		{

            el_at_pos_i_in_sigma_2 = indiv->genome[k];
		

            new_fitness_delta += m_distance_matrix[k][i] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[j]] - m_flow_matrix[indiv->genome[k]][indiv->genome[i]]);
            new_fitness_delta += m_distance_matrix[i][k] * (m_flow_matrix[indiv->genome[j]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[i]][indiv->genome[k]]);


            new_fitness_delta += m_distance_matrix[k][j] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[i]] - m_flow_matrix[indiv->genome[k]][indiv->genome[j]]);
            new_fitness_delta += m_distance_matrix[j][k] * (m_flow_matrix[indiv->genome[i]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[j]][indiv->genome[k]]);
		}

		int k = i;
        el_at_pos_i_in_sigma_2 = indiv->genome[j];
		new_fitness_delta += m_distance_matrix[k][i] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[j]] - m_flow_matrix[indiv->genome[k]][indiv->genome[i]]);
        new_fitness_delta += m_distance_matrix[i][k] * (m_flow_matrix[indiv->genome[j]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[i]][indiv->genome[k]]);

		k = j;
		el_at_pos_i_in_sigma_2 = indiv->genome[i];
		new_fitness_delta += m_distance_matrix[k][i] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[j]] - m_flow_matrix[indiv->genome[k]][indiv->genome[i]]);
		new_fitness_delta += m_distance_matrix[i][k] * (m_flow_matrix[indiv->genome[j]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[i]][indiv->genome[k]]);

		new_fitness_delta += m_distance_matrix[k][j] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[i]] - m_flow_matrix[indiv->genome[k]][indiv->genome[j]]);
		new_fitness_delta += m_distance_matrix[j][k] * (m_flow_matrix[indiv->genome[i]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[j]][indiv->genome[k]]);

		for (int k = i+1; k < j; k++)
		{

            el_at_pos_i_in_sigma_2 = indiv->genome[k];


            new_fitness_delta += m_distance_matrix[k][i] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[j]] - m_flow_matrix[indiv->genome[k]][indiv->genome[i]]);
            new_fitness_delta += m_distance_matrix[i][k] * (m_flow_matrix[indiv->genome[j]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[i]][indiv->genome[k]]);


            new_fitness_delta += m_distance_matrix[k][j] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[i]] - m_flow_matrix[indiv->genome[k]][indiv->genome[j]]);
            new_fitness_delta += m_distance_matrix[j][k] * (m_flow_matrix[indiv->genome[i]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[j]][indiv->genome[k]]);
		}

		for (int k = j+1; k < n; k++)
		{

                el_at_pos_i_in_sigma_2 = indiv->genome[k];
			

            new_fitness_delta += m_distance_matrix[k][i] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[j]] - m_flow_matrix[indiv->genome[k]][indiv->genome[i]]);
            new_fitness_delta += m_distance_matrix[i][k] * (m_flow_matrix[indiv->genome[j]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[i]][indiv->genome[k]]);


            new_fitness_delta += m_distance_matrix[k][j] * (m_flow_matrix[el_at_pos_i_in_sigma_2][indiv->genome[i]] - m_flow_matrix[indiv->genome[k]][indiv->genome[j]]);
            new_fitness_delta += m_distance_matrix[j][k] * (m_flow_matrix[indiv->genome[i]][el_at_pos_i_in_sigma_2] - m_flow_matrix[indiv->genome[j]][indiv->genome[k]]);
		}

		return -new_fitness_delta;
	}

	double QAP::fitness_delta_insert(CIndividual *indiv, int i, int j){
		InsertAt(indiv->genome, i, j, problem_size_PBP);
		double res = _Evaluate(indiv->genome);
		res -= indiv->f_value;
		InsertAt(indiv->genome, j, i, problem_size_PBP);
		return res;
	}
	


/*
 * Returns the size of the problem.
 */
int QAP::GetProblemSize()
{
    return n;
}

}