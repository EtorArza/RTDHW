#include "Tabu.h"
#include "Tools.h"
#include <limits>
#include "Parameters.h"
#include "constants.h"


namespace PERMU
{

Tabu::Tabu(RandomNumberGenerator *rng, int n)
{
	this->tabu_length = MIN_TABU_LENGTH;
	this->rng = rng;
	this->n = n;
	tabu_indices_i = new int[MAX_TABU_LENGTH];
	tabu_indices_j = new int[MAX_TABU_LENGTH];

	for (int k = 0; k < MAX_TABU_LENGTH; k++)
	{
		tabu_indices_i[k] = -1;
		tabu_indices_j[k] = -1;
	}

	tabu_table = new bool *[n];
	for (int i = 0; i < n; i++)
	{
		tabu_table[i] = new bool[n];
		for (int j = 0; j < n; j++)
		{
			tabu_table[i][j] = false;
		}
	}
	index_pos = 0;
}

Tabu::~Tabu()
{
	delete[] tabu_indices_i;
	delete[] tabu_indices_j;
	for (int i = 0; i < n; i++)
	{
		delete[] tabu_table[i];
	}
	delete[] tabu_table;
}

int Tabu::next_index_pos()
{
	index_pos++;
	index_pos = index_pos % tabu_length;
	return index_pos;
}

void Tabu::set_tabu(int i, int j)
{
	if (this->tabu_coef_neat < CUTOFF_0)
	{ // if tabu coef not high enough, do not add to tabu
		return;
	}
	if (i == -1 || j == -1)
	{
		return;
	}

	if (!(tabu_indices_i[index_pos] == -1))
	{
		tabu_table[tabu_indices_i[index_pos]][tabu_indices_j[index_pos]] = false;
	}
	tabu_indices_i[index_pos] = i;
	tabu_indices_j[index_pos] = j;
	tabu_table[i][j] = true;
	next_index_pos();
}

bool Tabu::is_tabu(int i, int j)
{
	// if tabu coef is near zero, return false, hence, not use tabu
	if (this->tabu_coef_neat < CUTOFF_0 && this->tabu_coef_neat > -CUTOFF_0)
	{
		return false;
	}

	if (i == -1 || j == -1)
	{
		return false;
	}

	return tabu_table[i][j];
}

void Tabu::reset()
{
	for (int k = 0; k < tabu_length; k++)
	{
		tabu_indices_i[k] = -1;
		tabu_indices_j[k] = -1;
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			tabu_table[i][j] = false;
		}
	}
	index_pos = 0;
	this->tabu_coef_neat = 0;
	this->tabu_length = MIN_TABU_LENGTH;
}

void Tabu::resize(int new_size)
{
	new_size = max(MIN_TABU_LENGTH, new_size);
	new_size = min(MAX_TABU_LENGTH, new_size);

	if (this->tabu_length != new_size)
	{
		this->tabu_length = new_size;
		double tmp_coef = this->tabu_coef_neat;
		index_pos = 0;
		for (int k = 0; k < tabu_length; k++)
		{
			tabu_indices_i[k] = -1;
			tabu_indices_j[k] = -1;
		}
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				tabu_table[i][j] = false;
			}
		}
	}
}

double Tabu::return_current_relative_tabu_size()
{
	return (double) (tabu_length - MIN_TABU_LENGTH) / (double) MAX_TABU_LENGTH;
}


void Tabu::increase_tabu_size()
{
	this->resize(this->tabu_length + LENGTH_CHANGE_STEP);
}
void Tabu::decrease_tabu_size()
{
	this->resize(this->tabu_length - LENGTH_CHANGE_STEP);
}


} // Namespace PERMU