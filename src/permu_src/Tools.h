#pragma once
/*
 *  Tools.h
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 11/21/11.
 *  Copyright 2011 University of the Basque Country. All rights reserved.
 *
 */
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <functional>
#include <mutex>
#include <iomanip>
#include "constants.h"

using std::istream;
using std::ostream;
using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::stringstream;


/*
 * Returs the first position at which value appears in array. If it does not appear, then it returns -1;
 */
int Find(int *array, int size, int value);

/*
 * Calculates Kullback-Leibeler divergence between P and Q distributions.
 */
double KullbackLeibelerDivergence(double *P, double *Q, int size);

/*
 * Calculates Total Variation divergence between P and Q distributions.
 */
double TotalVariationDivergence(double *P, double *Q, int size);

/*
 * It determines if the given int sequecen if it is indeed a permutation or not.
 */
bool isPermutation(int *permutation, int size);




/*
 * Determines if a given string contains a certain substring.
 */
bool strContains(const string inputStr, const string searchStr);

template <class T>
void PrintArray(T *array, int length)
{
    for (int i = 0; i < length; i++)
    {
        cout << std::setprecision(10);
        cout << array[i] << " ";
    }
    cout << " " << endl;
}

template <class T>
string array_to_python_list_string(T* array, int len)
{   
    if(len == 0)
    {
        return "[]";
    }

    stringstream res_ss;
    res_ss << "[" << array[0];
    for (size_t i = 1; i < len; i++)
    {
        res_ss << ", " << array[i];
    }
    res_ss << "]";
    return res_ss.str();
}



/*
 * Applies the random keys sorting strategy to the vector of doubles
 */
void RandomKeys(int *a, double *criteriaValues, int size);

/*
 * Calculates the tau Kendall distance between 2 permutations.
 */
int Kendall(int *permutationA, int *permutationB, int size);


double Average_Kendall_distance_between_random_permus(int n);

void generate_samples_kendall_distance_between_random_permus(int n);



/*
 * Calculates the Kendall tau distance between 2 permutations.
 */
int Kendall(int *permutationA, int *permutationB, int size, int *m_aux);

/*
 * Calculates the Kendall tau distance between 2 permutations.
 * Auxiliary parameters are used for multiple executions.
 */
int Kendall(int *permutationA, int *permutationB, int size, int *m_aux, int *invertedB, int *composition, int *v);

/*
 * Calculates the Cayley distance between 2 permutations.
 */
int Cayley(int *permutationA, int *permutationB, int size);

/*
 * Calculates the Cayley distance between 2 permutations.
 */
int Cayley(int *permutationA, int *permutationB, int size, int *invertedB, int *composition, int *elemsToCycles, int *maxPosInCycle, int *freeCycle);
int FindNewCycle(int *freeCycle, int size);
int NextUnasignedElem(int *elemsToCycles, int size);
int CalculateDistance(int *sigma, int size);

/*
 * Calculates the length of the longest increasing subsequence in the given array of ints.
 */
int getLISLength(int *sigma, int size);

/*
 * Implements the compose of 2 permutations of size n.
 */
void Compose(int *s1, int *s2, int *res, int n);

/*
* Calculates V_j-s vector.
*/
void vVector(int *v, int *permutation, int n);

/*
 *  Optimized version by Leti of the V_j-s vector calculation.
 */
void vVector_Fast(int *v, int *permutation, int n, int *m_aux);

/*
 * Inverts a permutation.
 */
void Invert(int *permu, int n, int *inverted);

/*
 * This method moves the value in position i to the position j.
 */
void InsertAt(int *array, int i, int j, int n);

/*
 * Calculates the factorial of a solution.
 */
long double factorial(int val);

/*
 * This method applies a swap of the given i,j positions in the array.
 */
void Swap(int *array, int i, int j);

// not copied the ones below this


/*
 * Calculate the Hamming distance between two permutations
 */

int Hamming_distance(int* sigma1, int* sigma2, int len);

/*
 * Set timer to 0.
 */
void tic();

/*
 * Return time since last tic().
 */
double toc();

/*
 * Convert to string.
 */
template <class T>
string toString(const T &t, bool *ok = NULL)
{
    ostringstream stream;
    stream << t;
    if (ok != NULL)
        *ok = (stream.fail() == false);
    return stream.str();
}

/*
 * Convert to string.
 * https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
 * sort 2 arrays simultaneously. The values used as keys are placed in A. If ascending == false, the order is descending.
 * default is descending.
 */
template <class A, class B>
void QuickSort2Desc(A a[], B b[], int l, int r, bool ascending=false)
{
    int i = l;
    int j = r;

    int coef = 1;
    if (ascending==true)
    {
        coef = -1;
    }

    A v = a[(l + r) / 2];

    
    do
    {
        while (coef*a[i] > coef*v)
            i++;
        while (coef*v > coef*a[j])
            j--;
        if (i <= j)
        {
            std::swap(a[i], a[j]);
            std::swap(b[i], b[j]);
            i++;
            j--;
        };
    } while (i <= j);
    if (l < j)
        QuickSort2Desc(a, b, l, j, ascending);
    if (i < r)
        QuickSort2Desc(a, b, i, r, ascending);
}

/*
 * Return wether two vectors are equal or not.
 */
template <class T>
bool compare_vectors(T *vec1, T *vec2, int len)
{
    for (int i = 0; i < len; i++)
    {
        if (vec1[i] != vec2[i])
        {
            return false;
        }
    }
    return true;
}

int count_n_dif_array_items_double(double* array1, double* array2, int n);
int count_n_dif_matrix_items_double(double** matrix1, double** matrix2, int n, int m);




/*
Function to find all the repeated rows on a matrix
Writes in  bool *is_ith_position_repeated (true --> vector is a repetition, false--> vector is not a repetition)
*/
template <class T>
void which_indexes_correspond_to_repeated_vectors(T **vec_array, int vec_len, int n_of_vecs, bool *is_ith_position_repeated, bool is_known_last_repeated_indexes)
{
    if (n_of_vecs == 1)
    {
        is_ith_position_repeated[0] = false;
        return;
    }
    else if (n_of_vecs == 2)
    {
        is_ith_position_repeated[0] = false;
        is_ith_position_repeated[1] = compare_vectors(vec_array[0], vec_array[1], vec_len);
        return;
    }

    is_ith_position_repeated[0] = false;
    is_ith_position_repeated[1] = compare_vectors(vec_array[0], vec_array[1], vec_len);

    for (int i = 2; i < n_of_vecs; i++)
    {
        if (is_known_last_repeated_indexes && not is_ith_position_repeated[i])
        {
            continue;
        }
        for (int j = i - 1; j >= 0; j--)
        {
            is_ith_position_repeated[i] = false;
            if (compare_vectors(vec_array[i], vec_array[j], vec_len))
            {
                is_ith_position_repeated[i] = true;
                break;
            }
        }
    }
}




// apply sigmoid function
double sigmoid(double x);



// round a double into the nearest integer
int tools_round(double x);

// compute the average value of the elements on the array
template <class T>
double Average(T *array, int len)
{

    double sum = 0;

    for (int i = 0; i < len; i++)
    {
        sum += array[i];
    }

    return (double)sum / len;
}


template <class A, class B>
void multiply_array_with_value(A* array, B value, int len){
    for (int i = 0; i < len; i++)
    {
        array[i] *= value;
    }
}

template <class T>
void sum_value_to_array(T* array, T value, int len){
    for (int i = 0; i < len; i++)
    {
        array[i] += value;
    }
}


template <class A, class B, class C>
void multiply_two_arrays_elementwise(A* array_1, B* array_2, C* res, int len){
    for (int i = 0; i < len; i++)
    {
        res[i] = array_1[i] * array_2[i];
    }
}


// Scalar multiplication of vectors: multiply elemntwise and sum them. The return type is the one on the first vector.
template <class A, class B>
A scalar_multiplication(A* array_1, B* array_2, int len){
    A res = 0;
    for (int i = 0; i < len; i++)
    {
        res += array_1[i] * array_2[i];
    }
    return res;
}
template<class T>
inline void set_array_to_value(T* array, T value, int length){
    for (int i = 0; i < length; i++)
    {
        array[i] = value;
    }
}

double euclid_dist(double* array_1, double* array_2, int len);

template <class T>
T l1_distance(T* array_1, T* array_2, int len)
{
    T res = 0;

    for (int i = 0; i < len; i++)
    {
        res += abs(array_1[i] - array_2[i]);
    }

    return res;
}

// compute the average value of the elements on the array dropping the best and worst quarter
template <class T>
double Average_drop_top_bottom_quartile(T *array, int len)
{

    if (len < 4)
    {
        return Average(array, len);
    }
    
    double sum = 0;
    std::sort(array,array+len);
    for (int i = len/4; i < len/4*3; i++)
    {
        sum += array[i];
    }

    return (double)sum / (len/4*3 - len/4);
}



template <class T>
T median(T *array, int len)
{

    std::nth_element(array, array + len/2, array+len);

    return *(array+len/2);

}


// compute the variance of the elements on the array
template <class T>
double Variance(T *array, int len)
{

    double mean = Average(array, len);

    double var = 0;
    for (int i = 0; i < len; i++)
    {
        var += (array[i] - mean) * (array[i] - mean);
    }

    return (double)var / len;
}

// Normalize a vector so that the sum of all the elements on it is 1
template <class T>
void normalize_vector(T *array, int len)
{
    T sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += array[i];
    }
    for (int i = 0; i < len; i++)
    {
        array[i] = array[i] / sum;
    }
}



template <class T>
void PrintMatrix(T **M, int m, int n)
{

	cout << "\n";
	for (int i = 0; i < m; i++)
	{
		cout << "| i = " << i << " ( ";
		for (int j = 0; j < n; j++)
		{
			cout << M[i][j] << " ";
		}
		cout << ")\n";
	}
}

template <class T>
void PrintMatrixVec(vector<vector<T>> M)
{
    int m = M.size();
    int n = M[0].size();
	cout << "\n";
	for (int i = 0; i < m; i++)
	{
		cout << "| i = " << i << " ( ";
		for (int j = 0; j < n; j++)
		{
			cout << M[i][j] << " ";
		}
		cout << ")\n";
	}
}

// Sum the elements from pos i to pos j-1. The python equivalent of sum(v[i:j])
template <class T>
T sum_slice_vec(T *v, int i, int j)
{
    T res = 0;
    for (int k = i; k < j; k++)
    {
        res += v[k];
    }
    return res;
}


// Sum the absolute value of real valued elements from pos i to pos j-1. The python equivalent of sum(v[i:j])
template <class T>
T sum_abs_val_slice_vec(T *v, int i, int j)
{
    T res = 0;
    for (int k = i; k < j; k++)
    {
        res += abs(v[k]);
    }
    return res;
}

template <class T>
int argmax(T *v, int len){
    T max = v[0];
    int res = 0;
    for (int i = 1; i < len; i++)
    {
        if (v[i] > max)
        {
            max = v[i];
            res = i;
        }
    }
    return res;
}

template <class T>
int move_to_0_minusone_or_one(T value)
{
    if ((double)value < -CUTOFF_0)
    {
        return -1;
    }
    else if ((double)value > CUTOFF_0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

template <class T>
int argmin(T *v, int len){
    T max = v[0];
    int res = 0;
    for (int i = 1; i < len; i++)
    {
        if (v[i] < max)
        {
            max = v[i];
            res = i;
        }
    }
    return res;
}


template <class T>
void copy_array(T *v_res, T*v_ref, int len){

    for (int i = 0; i < len; i++)
    {
        v_res[i] = v_ref[i];
    }
}

// https://thispointer.com/c-how-to-read-a-file-line-by-line-into-a-vector/
std::vector<string> read_lines_from_file(string filename);

std::vector<string> split(string txt, char ch);


inline
double fast_exp(double x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

class RandomNumberGenerator{

    public:

        unsigned long x, y, z;

        RandomNumberGenerator(){x=123456789, y=362436069, z=521288629;seed(2);}
        ~RandomNumberGenerator(){};
        void seed(void);
        void seed(int seed);

        std::vector<unsigned long> get_state();
        void set_state(std::vector<unsigned long> seed_state);
        int random_integer_fast(){return xorshf96();}
        int random_integer_fast(int max){return xorshf96() % max;}
        int random_integer_fast(int min, int max){return min + (xorshf96() % (max - min));}
        int random_integer_uniform(int max);
        int random_integer_uniform(int min, int max);
        double random_0_1_double();

        


        int xorshf96(void);


};


// Generate Random Permutation
void GenerateRandomPermutation(int *permutation, int n);
void GenerateRandomPermutation(int *permutation, int n, RandomNumberGenerator* rng);


// Generate real array of length n with numbers between 0 and 1
void GenerateRandomRealvec_0_1(double *real_vec, int n);
void GenerateRandomRealvec_0_1(double *real_vec, int n, RandomNumberGenerator* rng);


// Choose an index given the probabilities
int choose_index_given_probabilities(double *probabilities_array, int max_index);
int choose_index_given_probabilities(double *probabilities_array, int len, RandomNumberGenerator* rng);


// Choose an index given positive weights
int choose_index_given_weights(double *weights_array, int max_index);
int choose_index_given_weights(double *weights_array, int max_index, RandomNumberGenerator* rng);


// Sample from a bernouilli distribution.
bool coin_toss(double p_of_true);
bool coin_toss(double p_of_true, RandomNumberGenerator* rng);


void shuffle_vector(int *vec, int len);
void shuffle_vector(int *vec, int len, RandomNumberGenerator* rng);

class Lap;

namespace PERMU
{
class PermuTools
{
public:
    PermuTools(int n);
    PermuTools(int n, RandomNumberGenerator *rng);

    void init_class(int n);

    ~PermuTools();

    int n;
    int *random_permu1;
    int *random_permu2;
    int *temp_array;
    int *temp_array2;
    int *temp_array3;
    int *temp_array4;
    double *temp_array_double;
    int *identity_permu;

    int *hamming_mm_consensus;
    int *kendall_mm_consensus;

    double **first_marginal;
    double **order_marginal;

    void combine_permus(int **permu_list, double *coef_list, int *res);
    void compute_first_marginal(int **permu_list, int m);
    void compute_order_marginal(int **permu_list, int m);

    double get_distance_to_marginal(int *permu);
    double get_distance_to_order_marginal(int *permu);

    int choose_permu_index_to_move(double *coef_list);
    int choose_permu_index_to_move(double *coef_list, RandomNumberGenerator *rng);

    double compute_kendall_distance(int *premu_1, int *permu_2);
    void compute_kendall_consensus_borda(int **permu_list, int m);

    double compute_hamming_distance(int *permu_1, int *permu_2);
    void compute_hamming_consensus(int **permu_list, int m);

    double compute_hamming_distance_to_consensus(int *permu) { return compute_hamming_distance(permu, hamming_mm_consensus); }
    double compute_kendall_distance_to_consensus(int *permu) { return compute_kendall_distance(permu, kendall_mm_consensus); }

    double compute_normalized_hamming_distance_to_consensus(int *permu) { return compute_hamming_distance_to_consensus(permu) / (double)n; }
    double compute_normalized_kendall_distance_to_consensus(int *permu) { return compute_kendall_distance_to_consensus(permu) / (double)(n * (n - 1) / 2); }
    double compute_normalized_kendall_distance_to_consensus_fast_approx(int *permu) { return l1_distance(permu, kendall_mm_consensus, n) / (double)(n * n / 2); }


private:
    void convert_to_permu(int *res);
    RandomNumberGenerator *rng;

    Lap *linear_assigment_problem;

    int *lap_rows;
    int *lap_cols;
    int *lap_u;
    int *lap_v;

    int **freq_matrix;

    int it_to_compute_order_marginal = 0;

    bool delete_rng_is_required = false;
};
} // namespace PERMU

class stopwatch
{
public:
    stopwatch(){tt_tic = getTick();}
    ~stopwatch(){}
    void tic() { tt_tic = getTick();}
    double toc() { return (double) (getTick() - tt_tic);}
private:
    double getTick();
    double tt_tic;

};

template <class T>
T obtain_kth_smallest_value(T *v, int k, int len)
{
    if (k < 1 || k > len)
    {
        cout << "error, 0 < k < len + 1 must hold, k=" << k << " and len=" << len << " were provided. Exiting..." << endl;
        exit(1); 
    }
    
    T *v_copy = new T[len];
    memcpy(v_copy, v, sizeof(T)*len);
    std::nth_element(v_copy, v_copy + k -1, v_copy + len);
    T res = v_copy[k -1];
    delete[] v_copy;
    return res;
}

template <class T>
T obtain_kth_largest_value(T *v, int k, int len)
{
    return obtain_kth_smallest_value(v, (len - k + 1), len);
}


class _sort_indices
{
   private:
     int* mparr;
   public:
     _sort_indices(int* parr) : mparr(parr) {}
     bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
};


class _sort_indices_double
{
   private:
     double* mparr;
   public:
     _sort_indices_double(double* parr) : mparr(parr) {}
     bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
};

bool are_doubles_equal(double x1, double x2);

void compute_order_from_int_to_int(int* v, int len, int* order_res, bool reverse = false);

void compute_order_from_double_to_int(double* v, int len, int* order_res, bool reverse = false);

void compute_order_from_double_to_double(double* v, int len, double* order_res, bool reverse = false, bool respect_ties = false);



// if NOT reverse, then the smallest value (0.0 in the case of the normalized result) will NOT change position
void transform_from_values_to_normalized_rankings(double* reference_and_result, int len, bool reverse = false);

// https://www.geeksforgeeks.org/rounding-floating-point-number-two-decimal-places-c-c/
double tools_round_two_decimals(double x);


void transform_from_values_to_geometric_ranking_probs(double* reference_and_result, int len, bool reverse = false);




// return element in position specified by percentage. For example, percentage=0.25 selects the first element in the first quartile
// 0.0 selects the largest value in the array.
template <class T>
int arg_element_in_centile_specified_by_percentage(T* array, int len, double percentage){
    
    int index = tools_round((double) (len - 1) * percentage);


    T value = obtain_kth_largest_value(array, index + 1, len);

    for (int i = 0; i < len; i++)
    {
        if(value == array[i]){
            return i;
        }
    }

    cout << "error, item not found in function \"arg_element_in_centile_specified_by_percentage\". Exit..." << endl;
    exit(1);

}


//Unpaired test
bool is_A_larger_than_B_Mann_Whitney(double* A, double* B, int length, int ALPHA_INDEX);

//Paired test
bool is_A_larger_than_B_Signed_Wilcoxon(double* A, double* B, int length, int ALPHA_INDEX);

//Paired test
bool Friedman_test_are_there_critical_diferences(double** f_values, int n_candidates, int n_samples, int ALPHA_INDEX);

void get_ranks_from_f_values(vector<vector<double>>& ranks, double** f_values, int n_candidates, int n_samples);

double p_value_chisquared(double x, double df);


void F_race_iteration(double** f_values, vector<int> &surviving_candidates, int n_samples, int ALPHA_INDEX);


template <class T>
int sign(T value)
{
    if (value < SMALLEST_POSITIVE_DOUBLE && -SMALLEST_POSITIVE_DOUBLE < value)
    {
        return 0;
    }
    else if (value > 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}


template <class T>
void append_line_to_file(std::string filename, T data_to_append ) {  
  std::ofstream outfile;
  outfile.open(filename, std::ios_base::app); // append instead of overwrite
  outfile << data_to_append;
  outfile.flush();
  outfile.close();
}


// sum two arrays of the same length elementwise. The result array is allowed to be one of the summands (although it is not required)
template <class T>
void sum_arrays(T* res, T* array_1, T* array_2, int len){
    for (int i = 0; i < len; i++)
    {
        res[i] = array_1[i] + array_2[i];
    }
}



template <class T>
std::string array_to_string(T* array, int len){
    std::string res = "[";
    for (int i = 0; i < len; i++)
    {
        if(i != 0){
            res += ",";
        }
        res += to_string(array[i]);
    }
    res += "]";
    return res;
}




std::string from_path_to_filename(std::string file_path);

//https://stackoverflow.com/questions/52206675/template-print-function-c
template <typename TType>
void print_vector(const std::vector<TType>& vec)
{
    typename  std::vector<TType>::const_iterator it;
    std::cout << "{";
    for(it = vec.begin(); it != vec.end(); it++)
    {
        if(it!= vec.begin()) std::cout << ", ";
        std::cout << (*it);
    }
    std::cout << "}";
}

//https://stackoverflow.com/questions/52206675/template-print-function-c
template <typename T2>
void print_vector(const std::vector< std::vector<T2> >& vec)
{
    for( auto it= vec.begin(); it!= vec.end(); it++)
    {
        print_vector(*it);
    }
}


template <typename TType>
bool vector_contains_item(const std::vector<TType>& vec, TType item)
{
	if (std::count(vec.begin(), vec.end(), item))
		{return true;}
	else
		{return false;}
}

template <class T>
void find_classes_in_array_of_objects(T *array_of_objects, std::function<bool(T, T)> are_equal, int len, int* classes_array)
{
    for (int i = 0; i < len; i++)
    {
        classes_array[i] = -1;
    }
    for (int i = 0; i < len - 1; i++)
    {
        if (classes_array[i] != -1)
        {
            continue;
        }
        for (int j = i + 1; j < len; j++)
        {
            if (classes_array[j] != -1)
            {
                continue;
            }
            if (are_equal(array_of_objects[i], array_of_objects[j]))
            {
                classes_array[j] = i;
            }
        }
    }

    int repetitions = 0;
    for (int i = 0; i < len; i++)
    {
        if (classes_array[i] == -1)
        {
            classes_array[i] = i;
        }else{
            repetitions += 1;
        }

    }
    
    cout << "Repeated: " << repetitions;
    //PrintArray(classes_array, len);
}

template< class T>
int index(const T* array, const T item, int len){
    for (int i = 0; i < len; i++)
    {
        if (array[i] == item)
        {
            return i;
        }
    }
    return -1;
}

template< class T>
bool is_item_in_array(const T* array, const T item, int len){
    for (int i = 0; i < len; i++)
    {
        if (array[i] == item)
        {
            return true;
        }
    }
    return false;
}


int n_choose_k(int n, int k);

class progress_bar
{

public:

int max_steps;
int current_steps;
bool printed_bracket = false;
stopwatch timer;

progress_bar(int n);
~progress_bar();
void step();
void end();
void restart(int n);


};


vector<string> split_string(string str, string token);


std::string system_exec(const char* cmd);

std::string system_exec(std::string cmd);


// get runtime of job in hipatia to get around the suspension mechanism during training
double get_runtime_hipatia();


void mkdir(const string &path);
bool exists(const std::string &path);