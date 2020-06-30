/*
 *  TSP.cpp
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 7/11/13.
 *  Copyright 2013 University of the Basque Country. All rights reserved.
 *
 */

#include "TSP.h"
#include <assert.h>

namespace PERMU{

/*
 *Class constructor.
 */
TSP::TSP()
{
}

/*
 * Class destructor.
 */
TSP::~TSP()
{
	for (int i = 0; i < m_size; i++)
		delete[] m_distance_matrix[i];
	delete[] m_distance_matrix;
}

double CalculateGEODistance(double latitudeX, double latitudeY, double longitudeX, double longitudeY)
{
	double PI = 3.141592;
	double RRR = 6378.388;

	double deg = (double)((int)latitudeX);
	double min = latitudeX - deg;
	double latitudeI = PI * (deg + 5.0 * min / 3.0) / 180.0;

	deg = (double)((int)latitudeY);
	min = latitudeY - deg;
	double longitudeI = PI * (deg + 5.0 * min / 3.0) / 180.0;

	deg = (double)((int)longitudeX);
	min = longitudeX - deg;
	double latitudeJ = PI * (deg + 5.0 * min / 3.0) / 180.0;

	deg = (double)((int)longitudeY);
	min = longitudeY - deg;
	double longitudeJ = PI * (deg + 5.0 * min / 3.0) / 180.0;

	double q1 = cos(longitudeI - longitudeJ);
	double q2 = cos(latitudeI - latitudeJ);
	double q3 = cos(latitudeI + latitudeJ);

	return (int)(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
}

/*
 * Read TSP instance file that belongs to the TSPLIB library.
 */
int TSP::Read2(string filename)
{
	//declaration and initialization of variables.
	bool readm_distance_matrix = false;
	bool coordinatesData = false;
	string DISTANCE_TYPE;
	double **coordinates;
	char line[5048]; // variable for input value
	string data = "";
	ifstream indata;
	indata.open(filename.c_str(), ios::in);

	while (!indata.eof())
	{
		//LEER LA LINEA DEL FICHERO
		indata.getline(line, 5048);
		stringstream ss;
		string sline;
		ss << line;
		ss >> sline;
		if (sline == "EOF")
		{
			break;
		}
		if (readm_distance_matrix && coordinatesData == false)
		{
			//cout << "reading distance m_distance_matrix "<<line<< endl;
			if (data == "")
				data = line;
			else
				data = data + ' ' + line;
		}
		else if (readm_distance_matrix && coordinatesData == true)
		{
			//FILL DISTANCE m_distance_matrix
			char *coordPieces;
			coordPieces = strtok(line, " ");
			if (strcmp(coordPieces, " ") == 0)
			{
				coordPieces = strtok(NULL, " ");
			}
			int coordNum = atoi(coordPieces);
			coordPieces = strtok(NULL, " ");
			double latitud = atof(coordPieces);
			coordPieces = strtok(NULL, " ");
			double longitud = atof(coordPieces);
			double *coordinate = new double[2];
			coordinate[0] = latitud;
			coordinate[1] = longitud;

			coordinates[coordNum - 1] = coordinate;
			//cout<<"coordNum "<<coordNum-1<<" latit: "<<latitud<<" long: "<<longitud<<endl;
		}

		if (strContains(sline, "DIMENSION"))
		{
			char *pch;
			pch = strtok(line, " ");
			pch = strtok(NULL, " ");
			if (strcmp(pch, ":") == 0)
			{
				pch = strtok(NULL, " ");
			}
			m_size = atoi(pch);
		}
		else if (strContains(sline, "EDGE_WEIGHT_TYPE"))
		{
			char *pch;
			pch = strtok(line, " ");
			pch = strtok(NULL, " ");
			if (strcmp(pch, ":") == 0)
			{
				pch = strtok(NULL, " ");
			}
			stringstream s;
			string type;
			s << pch;
			s >> type;
			DISTANCE_TYPE = type;
		}
		else if (sline == "EDGE_WEIGHT_SECTION")
		{
			readm_distance_matrix = true;
			coordinatesData = false;
		}
		else if (sline == "NODE_COORD_SECTION")
		{
			readm_distance_matrix = true;
			coordinatesData = true;
			coordinates = new double *[m_size];
		}
	}
	indata.close();

	//BUILD DISTANCE m_distance_matrix
	m_distance_matrix = new double *[m_size];
	for (int i = 0; i < m_size; i++)
	{
		m_distance_matrix[i] = new double[m_size];
	}


	//FILL DISTANCE m_distance_matrix
	if (coordinatesData == true)
	{
		//CALCULATE EUCLIDEAN DISTANCES
		for (int i = 0; i < m_size; i++)
		{
			//get coordinate A
			double *coordA = coordinates[i];
			double coordAx = coordA[0];
			double coordAy = coordA[1];
			for (int j = i; j < m_size; j++)
			{
				//get coordinate B.
				double *coordB = coordinates[j];
				double coordBx = coordB[0];
				double coordBy = coordB[1];
				double euclidean;
				if (DISTANCE_TYPE == "GEO")
				{
					//calculate geographic distance between A and B.
					euclidean = CalculateGEODistance(coordAx, coordAy, coordBx, coordBy);
				}
				else
				{
					//calculate euclidean distance between A and B.
					double absolute = (coordAx - coordBx)*(coordAx - coordBx) + (coordAy - coordBy)*(coordAy - coordBy);
					euclidean = sqrt(absolute);
				}
				m_distance_matrix[i][j] = euclidean;
				m_distance_matrix[j][i] = euclidean; //<-symmetric m_distance_matrix
			}
		}
	}
	else
	{
		//FILL DISTANCE m_distance_matrix
		istringstream iss(data);
		int i = 0;
		int j = 0;
		do
		{
			string sub;
			iss >> sub;
			if (sub != "")
			{
				//save distance in distances m_distance_matrix. Save negative distance in order to minimize fitness instead of
				//maximize.
				m_distance_matrix[i][j] = atoi(sub.c_str());
				m_distance_matrix[j][i] = atoi(sub.c_str()); //<-symmetric m_distance_matrix
				if (sub == "0")
				{
					i++;
					j = 0;
				}
				else
				{
					j++;
				}
			}
			else
			{
				break;
			}
		} while (iss);
	}

	initialize_variables_PBP(m_size);

	return (m_size);
}

int TSP::Read(string filename)
{
	return Read2(filename);
}

// /*
//  * Read TSP instance file.
//  */
// int TSP::Read(string filename)
// {
// 	char line[2048]; // variable for input value
// 	string data="";
// 	ifstream indata;
// 	indata.open(filename.c_str(),ios::in);
// 	int num=0;
// 	while (!indata.eof())
// 	{
// 		indata.getline(line, 2048);
// 		stringstream ss;
// 		string sline;
// 		ss << line;
// 		ss >> sline;
// 		if (sline=="")
// 		{
// 			break;
// 		}
// 		if (num==0)
// 		{
// 			m_size = atoi(line);
// 		}
// 		else
// 		{
// 			if (data=="")
// 				data = line;
// 			else
// 				data = data+' '+line;
// 		}
// 		num++;
// 	}
// 	indata.close();

// 	//BUILD MATRIX
// 	m_distance_matrix = new double*[m_size];
// 	for (int i=0;i<m_size;i++)
// 	{
// 		m_distance_matrix[i]= new double[m_size];
// 	}

// 	istringstream iss(data);
// 	int i=0;
// 	int j=0;
// 	do
// 	{
// 		string sub;
// 	    iss >> sub;
// 	    if (sub!=""){
// 			//save distance in distances matrix.
// 	    	m_distance_matrix[i][j]= atoi(sub.c_str());
// 	    	if (j==(m_size-1))
// 	    	{
// 	    		i++;
// 	    		j=0;
// 	    	}
// 	    	else
// 	    	{
// 	    		j++;
// 	    	}
// 	    }
// 	    else
// 	    {
// 	    	break;
// 	    }
// 	} while (iss);
// 	initialize_variables_PBP(m_size);

// 	return (m_size);
// }

/*
 * This function evaluates the fitness of the solution for the TSP problem.
 */
double TSP::_Evaluate(int *genes)
{
	double distanceSum = 0;
	double distAB = 0;
	int IDCityA, IDCityB;
	for (int i = 0; i < m_size; i++)
	{
		IDCityA = genes[i];
		IDCityB = genes[0];
		if (i + 1 < m_size)
		{
			IDCityB = genes[i + 1];
		}

		distAB = m_distance_matrix[IDCityA][IDCityB];
		distanceSum = distanceSum + distAB;

		//security condition
		if (IDCityA == m_size || IDCityB == m_size)
		{
			distanceSum = 0;
			break;
		}
	}
	return -distanceSum;
}

double TSP::fitness_delta_swap(CIndividual *indiv, int i, int j)
{

	double delta = 0;
	assert(i + 1 == j || (i == 0 && j == problem_size_PBP - 1));

	if (i==0 && j==(problem_size_PBP-1)){
		i = problem_size_PBP-1;
		j = 0;
	}

	int i_prev = (i - 1);
	int j_next = (j + 1);


	if (i == 0)
	{
		i_prev = this->problem_size_PBP - 1;
	}

	if(j == this->problem_size_PBP - 1)
	{
		j_next = 0;
	}
	




	delta -= m_distance_matrix[indiv->genome[i_prev]][indiv->genome[i]];
	delta -= m_distance_matrix[indiv->genome[j]][indiv->genome[j_next]];

	delta += m_distance_matrix[indiv->genome[i_prev]][indiv->genome[j]];
	delta += m_distance_matrix[indiv->genome[i]][indiv->genome[j_next]];

	delta -= m_distance_matrix[indiv->genome[i]][indiv->genome[j]];
	delta += m_distance_matrix[indiv->genome[j]][indiv->genome[i]];

	return -delta;
}

double TSP::fitness_delta_interchange(CIndividual *indiv, int i, int j)
{

	if (i == j)
	{
		return 0;
	}
	else if (i > j)
	{


		return fitness_delta_interchange(indiv, j, i);
	}else if(i + 1 == j){


		return fitness_delta_swap(indiv, i, j);
	}else if(i == 0 && j == problem_size_PBP - 1){


		return fitness_delta_swap(indiv, i, j);

	}

	double delta = 0;

	

	int i_prev = (i - 1);
	int j_next = (j + 1);

	if (i == 0)
	{
		i_prev = this->problem_size_PBP - 1;
	}

	if(j == this->problem_size_PBP - 1)
	{
		j_next = 0;
	}



	delta -= m_distance_matrix[indiv->genome[i_prev]][indiv->genome[i]];
	delta -= m_distance_matrix[indiv->genome[i]][indiv->genome[i + 1]];
	delta -= m_distance_matrix[indiv->genome[j - 1]][indiv->genome[j]];
	delta -= m_distance_matrix[indiv->genome[j]][indiv->genome[j_next]];

	delta += m_distance_matrix[indiv->genome[i_prev]][indiv->genome[j]];
	delta += m_distance_matrix[indiv->genome[j]][indiv->genome[i + 1]];
	delta += m_distance_matrix[indiv->genome[j - 1]][indiv->genome[i]];
	delta += m_distance_matrix[indiv->genome[i]][indiv->genome[j_next]];


	return -delta;
}

double TSP::fitness_delta_insert(CIndividual *indiv, int i, int j)
{

	// if insertion occurs with respect to the first one and the last one, the objective value does not change
	// Only the representation shifts one to the left or to the right.
	if((i == 0 && j == problem_size_PBP - 1) || (j == 0 && i == problem_size_PBP - 1)){
		return 0.0;
	}


	int i_prev = (i - 1);
	int j_prev = (j - 1);
	int i_next = (i + 1);
	int j_next = (j + 1);


	if (i == 0)
	{
		i_prev = this->problem_size_PBP - 1;
	}else if (i == this->problem_size_PBP - 1)
	{
		i_next = 0;
	}
	

	if (j == 0)
	{
		j_prev = this->problem_size_PBP - 1;
	}else if (j == this->problem_size_PBP - 1)
	{
		j_next = 0;
	}


	if (i < j)
	{
		double delta = 0;

		delta -= m_distance_matrix[indiv->genome[i_prev]][indiv->genome[i]];
		delta -= m_distance_matrix[indiv->genome[i]][indiv->genome[i + 1]];
		delta -= m_distance_matrix[indiv->genome[j]][indiv->genome[j_next]];

		delta += m_distance_matrix[indiv->genome[i_prev]][indiv->genome[i + 1]];
		delta += m_distance_matrix[indiv->genome[j]][indiv->genome[i]];
		delta += m_distance_matrix[indiv->genome[i]][indiv->genome[j_next]];

		return -delta;
	}
	else if (j < i)
	{
		double delta = 0;

		delta -= m_distance_matrix[indiv->genome[j_prev]][indiv->genome[j]];
		delta -= m_distance_matrix[indiv->genome[i - 1]][indiv->genome[i]];
		delta -= m_distance_matrix[indiv->genome[i]][indiv->genome[i_next]];

		delta += m_distance_matrix[indiv->genome[j_prev]][indiv->genome[i]];
		delta += m_distance_matrix[indiv->genome[i]][indiv->genome[j]];
		delta += m_distance_matrix[indiv->genome[i - 1]][indiv->genome[i_next]];

		return -delta;
	}
	else
	{
		return 0;
	}
}

// Returns the size of the problem.
int TSP::GetProblemSize()
{
	return m_size;
}

}