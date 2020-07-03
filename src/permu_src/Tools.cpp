/*
 *  Tools.cpp
 *  RankingEDAsCEC
 *
 *  Created by Josu Ceberio Uribe on 11/21/11.
 *  Copyright 2011 University of the Basque Country. All rights reserved.
 *
 */

#include "Tools.h"
#include <limits.h>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h> /* srand, rand */
#include <assert.h>
#include "Parameters.h"
#include <set>
#include "permuevaluator.h"
#include "Lap.h"
#include <float.h>
#include <vector>
#include "constants.h"
#include <sys/types.h>
#include <sys/stat.h>



/*
 * Returs the first position at which value appears in array. If it does not appear, then it returns -1;
 */
int Find(int *array, int size, int value)
{
    int i = 0;
    while (i < size)
    {
        if (array[i] == value)
            return i;
        i++;
    }
    return -1;
}

/*
 * Calculates Kullback-Leibeler divergence between P and Q distributions.
 */
double KullbackLeibelerDivergence(double *P, double *Q, int SearchSpaceSize)
{
    double divergence = 0;
    double auxi = 0;
    for (int i = 0; i < SearchSpaceSize; i++)
    {
        auxi = P[i] * (log(P[i] / Q[i]));
        divergence = divergence + auxi;
    }
    return divergence;
}

/*
 * Calculates Total Variation divergence between P and Q distributions.
 */
double TotalVariationDivergence(double *P, double *Q, int size)
{
    double divergence = 0;
    for (int i = 0; i < size; i++)
    {
        divergence += fabs(P[i] - Q[i]);
    }
    return (0.5 * divergence);
}

/*
 * It determines if the given int sequecen if it is indeed a permutation or not.
 */
bool isPermutation(int *permutation, int size)
{
    int *flags = new int[size];
    for (int i = 0; i < size; i++)
        flags[i] = 1;

    for (int i = 0; i < size; i++)
    {
        int value = permutation[i];
        flags[value] = 0;
    }

    int result, sum = 0;
    for (int i = 0; i < size; i++)
        sum += flags[i];
    if (sum == 0)
        result = true;
    else
        result = false;
    delete[] flags;
    return result;
}



/*
 * Determines if a given string contains a certain substring.
 */
bool strContains(const string inputStr, const string searchStr)
{
    size_t contains;

    contains = inputStr.find(searchStr);

    if (contains != string::npos)
        return true;
    else
        return false;
}



/*
 * Prints in standard output 'length' double elements of a given array.
 */
void PrintArray(float *array, int length)
{
    int i;
    for (i = 0; i < length; i++)
        printf("%3.5f,", array[i]);
    std::cout << std::endl;
}



void PrintArray(double *array, int length){
    int i;
    for (i = 0; i < length; i++)
        printf("%3.5f,", array[i]);
    std::cout << std::endl;
}


/*
 * Prints in standard output 'lengthxlength' double elements of a given matrix.
 */
void PrintMatrixDouble(double **matrix, int length, int length2, string text)
{
    int i, j;
    cout << text << endl;
    for (i = 0; i < length; i++)
    {
        for (j = 0; j < length2; j++)
            printf("%3.9f, ", matrix[i][j]);
        printf("\n");
    }
}

/*
 * Calculates the tau Kendall distance between 2 permutations.
 */
int Kendall(int *permutationA, int *permutationB, int size)
{
    int i, dist;
    int *v, *composition, *invertedB, *m_aux;

    dist = 0;
    v = new int[size - 1];
    composition = new int[size];
    invertedB = new int[size];
    m_aux = new int[size];

    Invert(permutationB, size, invertedB);
    Compose(permutationA, invertedB, composition, size);
    vVector_Fast(v, composition, size, m_aux);

    for (i = 0; i < size - 1; i++)
        dist += v[i];

    delete[] composition;
    delete[] invertedB;
    delete[] v;
    delete[] m_aux;
    if (dist < 0)
    {
        cerr << "Kendall distance should be positive.";
        exit(1);
    }
    
    return dist;
}


double Average_Kendall_distance_between_random_permus(int n)
{
    double res = 0.0;
    int REPETITIONS = 100000;
    int* permu1 = new int[n];
    int* permu2 = new int[n];

    for (int i = 0; i < REPETITIONS; i++)
    {
        GenerateRandomPermutation(permu1, n);
        GenerateRandomPermutation(permu2, n);
        res += (double) Kendall(permu1, permu2, n);
    }
    res /= (double) REPETITIONS;
    return res;
}


void generate_samples_kendall_distance_between_random_permus(int n)
{
    double res = 0.0;
    int REPETITIONS = 100000;
    int* permu1 = new int[n];
    int* permu2 = new int[n];
    double max_kendall_dist = (double) n_choose_k(n,2);
    string res_str = "";

    for (int i = 0; i < REPETITIONS; i++)
    {
        GenerateRandomPermutation(permu1, n);
        GenerateRandomPermutation(permu2, n);
        res_str += to_string((double) Kendall(permu1, permu2, n) / max_kendall_dist);
        if (i == REPETITIONS -1)
        {
            break;
        }
        res_str += ", ";
    }
    res_str += "\n";
    append_line_to_file("random_permus_kendall_distance.txt", res_str);
    exit(1);
    res /= (double) REPETITIONS;
}



/*
 * Calculates the tau Kendall distance between 2 permutations.
 */
int Kendall(int *permutationA, int *permutationB, int size, int *m_aux)
{
    int i, dist;
    int *v, *composition, *invertedB;

    dist = 0;
    v = new int[size - 1];
    composition = new int[size];
    invertedB = new int[size];

    Invert(permutationB, size, invertedB);
    Compose(permutationA, invertedB, composition, size);
    vVector_Fast(v, composition, size, m_aux);

    for (i = 0; i < size - 1; i++)
        dist += v[i];

    delete[] composition;
    delete[] invertedB;
    delete[] v;

    return dist;
}

/*
 * Calculates the Kendall tau distance between 2 permutations.
 * Auxiliary parameters are used for multiple continuous executions.
 */
int Kendall(int *permutationA, int *permutationB, int size, int *m_aux, int *invertedB, int *composition, int *v)
{
    int i, dist;
    dist = 0;
    Invert(permutationB, size, invertedB);
    Compose(permutationA, invertedB, composition, size);
    vVector_Fast(v, composition, size, m_aux);
    for (i = 0; i < size - 1; i++)
        dist += v[i];

    return dist;
}

/*
 * Calculates the Cayley distance between 2 permutations.
 */
int Cayley(int *permutationA, int *permutationB, int size)
{
    int *invertedB = new int[size];
    int *composition = new int[size];
    int *elemsToCycles = new int[size];
    int *maxPosInCycle = new int[size];
    int *freeCycle = new int[size];

    Invert(permutationB, size, invertedB);
    Compose(permutationA, invertedB, composition, size);

    int index, cycle, distance;

    for (int i = 0; i < size; i++)
    {
        elemsToCycles[i] = -1;
        maxPosInCycle[i] = -1;
        freeCycle[i] = 1;
    }

    while ((index = NextUnasignedElem(elemsToCycles, size)) != -1)
    {
        cycle = FindNewCycle(freeCycle, size);
        freeCycle[cycle] = 0;
        do
        {
            elemsToCycles[index] = cycle;
            index = composition[index]; //para permus de 1..n =>index = sigma[index]-1;
        } while (elemsToCycles[index] == -1);
    }
    distance = size - FindNewCycle(freeCycle, size);

    delete[] invertedB;
    delete[] composition;
    delete[] elemsToCycles;
    delete[] maxPosInCycle;
    delete[] freeCycle;

    return distance;
}

/*
 * Calculates the Cayley distance between 2 permutations.
 */
int Cayley(int *permutationA, int *permutationB, int size, int *invertedB, int *composition, int *elemsToCycles, int *maxPosInCycle, int *freeCycle)
{

    Invert(permutationB, size, invertedB);
    Compose(permutationA, invertedB, composition, size);

    int index, cycle, distance;

    for (int i = 0; i < size; i++)
    {
        elemsToCycles[i] = -1;
        maxPosInCycle[i] = -1;
        freeCycle[i] = 1;
    }

    while ((index = NextUnasignedElem(elemsToCycles, size)) != -1)
    {
        cycle = FindNewCycle(freeCycle, size);
        freeCycle[cycle] = 0;
        do
        {
            elemsToCycles[index] = cycle;
            index = composition[index]; //para permus de 1..n =>index = sigma[index]-1;
        } while (elemsToCycles[index] == -1);
    }
    distance = size - FindNewCycle(freeCycle, size);

    return distance;
}

int Hamming_distance(int *sigma1, int *sigma2, int len)
{
    int res = 0;
    for (int i = 0; i < len; i++)
    {
        if (sigma1[i] != sigma2[i])
        {
            res++;
        }
    }
    return res;
}

int FindNewCycle(int *freeCycle, int size)
{
    int i;

    for (i = 0; i < size; i++)
        if (freeCycle[i])
            return i;
    return size;
}

int NextUnasignedElem(int *elemsToCycles, int size)
{
    int i;
    for (i = 0; i < size; i++)
        if (elemsToCycles[i] == -1)
            return i;
    return -1;
}

/*
 * Calculates the length of the longest increasing subsequence in the given array of ints.
 */
int getLISLength(int *sigma, int size)
{

    // O(n log k)

    int i;
    vector<int> vc(1, sigma[0]);
    vector<int>::iterator vk;

    for (i = 1; i < size; i++)
    {
        for (vk = vc.begin(); vk != vc.end(); vk++)
            if (*vk >= sigma[i])
                break;
        if (vk == vc.end())
            vc.push_back(sigma[i]);
        else
            *vk = sigma[i];
    }

    return (int)vc.size();
}
/*
 * Implements the compose of 2 permutations of size n.
 */
void Compose(int *s1, int *s2, int *res, int n)
{
    int i;
    for (i = 0; i < n; i++)
        res[i] = s1[s2[i]];
}

/*
 * Calculates V_j-s vector.
 */
void vVector(int *v, int *permutation, int n)
{

    int i, j;
    for (i = 0; i < n - 1; i++)
        v[i] = 0;

    for (i = n - 2; i >= 0; i--)
        for (j = i + 1; j < n; j++)
            if (permutation[i] > permutation[j])
                v[i]++;
}

/*
 *  Optimized version proposed by Leti for the calculation of the V_j-s vector.
 */
void vVector_Fast(int *v, int *permutation, int n, int *m_aux)
{
    int i, j, index;

    for (i = 0; i < n - 1; i++)
    {
        v[i] = 0;
        m_aux[i] = 0;
    }
    m_aux[n - 1] = 0;
    for (j = 0; j < n - 1; j++)
    {
        index = permutation[j];
        v[j] = index - m_aux[index];
        for (i = index; i < n; i++)
            m_aux[i]++;
    }
}

/*
 * Inverts a permutation.
 */
void Invert(int *permu, int n, int *inverted)
{
    int i;
    for (i = 0; i < n; i++)
        inverted[permu[i]] = i;
}

/*
 * Applies the random keys sorting strategy to the vector of doubles
 */
void RandomKeys(int *a, double *criteriaValues, int size)
{
    bool *fixedValues = new bool[size];
    double criteria, min;
    int i, j;
    for (i = 0; i < size; i++)
    {
        fixedValues[i] = false;
        a[i] = 0;
    }
    int minPos = 0;
    for (i = 0; i < size; i++)
    {
        min = 10000000;
        for (j = 0; j < size; j++)
        {
            criteria = criteriaValues[j];
            if (!fixedValues[j] && min > criteria)
            {
                min = criteria;
                minPos = j;
            }
        }

        fixedValues[minPos] = true;
        //a[i]=minPos;// modification por el asunto ordering /ranking
        a[minPos] = i; // original.
    }
    delete[] fixedValues;
}

/*
 * This method moves the value in position i to the position j.
 */
void InsertAt(int *array, int i, int j, int n)
{
    if (i != j)
    {
        int *res = new int[n];
        int val = array[i];
        if (i < j)
        {
            memcpy(res, array, sizeof(int) * i);

            for (int k = i + 1; k <= j; k++)
                res[k - 1] = array[k];

            res[j] = val;

            for (int k = j + 1; k < n; k++)
                res[k] = array[k];
        }
        else if (i > j)
        {
            memcpy(res, array, sizeof(int) * j);

            res[j] = val;

            for (int k = j; k < i; k++)
                res[k + 1] = array[k];

            for (int k = i + 1; k < n; k++)
                res[k] = array[k];
        }
        memcpy(array, res, sizeof(int) * n);
        delete[] res;
    }
}

/*
 * Calculates the factorial of a solution.
 */
long double factorial(int val)
{
    if (val <= 0)
        return 1;
    //long  N, b, c, p; // use int for fast calculation and small range of calculation..
    long b, c;
    long double p, N;
    N = (long double)val;
    c = (long)N - 1;
    p = 1;
    while (c > 0)
    {
        p = 0;
        b = c;
        while (b > 0)
        {
            if (b & 1)
            {
                p += N; // p = p + N;
            }
            // if you would like to use double choose the alternative forms instead shifts
            // the code is fast even!
            // you can use the same tips on double or 64 bit int etc.... but you must... ;-)
            //b >>= 1; // b/=2; (b = b / 2;) ( b >> 1; a.s.r. is more efficent for int or long..!)
            b /= 2;
            //N <<= 1; // N += N; N = N + N; N = N * 2; (N <<=1; a.s.l. is more efficent for int or long..!)
            N += N;
        } // end of: while(b>0)
        N = p;
        c--; // c = c - 1;
    }        // end of: while(c > 0)
    //printf("[%d] is the factorial! \n", p);
    return p;
}

/*
 * This method applies a swap of the given i,j positions in the array.
 */
void Swap(int *array, int i, int j)
{
    int aux = array[i];
    array[i] = array[j];
    array[j] = aux;
}



double stopwatch::getTick()
{ 
    struct timespec ts;
    double theTick;
    clock_gettime(CLOCK_REALTIME, &ts);
    theTick = (double)ts.tv_nsec / 1000000000.0;
    theTick += (double)ts.tv_sec;
    return theTick;
}


int RandomNumberGenerator::xorshf96(void)
{ //period 2^96-1
    unsigned long t;
    this->x ^= this->x << 16;
    this->x ^= this->x >> 5;
    this->x ^= this->x << 1;

    t = this->x;
    this->x = this->y;
    this->y = this->z;
    this->z = t ^ this->x ^ this->y;
    int res = abs((int) this->z % INT_MAX);
    return res;
}




void RandomNumberGenerator::seed(void){
    cout << "seed(void) called in Tools.cpp" << endl;
    exit(1);
}

void RandomNumberGenerator::seed(int seed){
    this->x = (unsigned long) seed;
    this->y=362436069UL; 
    this->z=521288629UL;
    xorshf96();
}


std::vector<unsigned long> RandomNumberGenerator::get_state(){
    std::vector<unsigned long> res = {x, y, z};
    return res;
}


void RandomNumberGenerator::set_state(std::vector<unsigned long> seed_state){
    x = seed_state[0];
    y = seed_state[1];
    z = seed_state[2];
}



// https://ericlippert.com/2013/12/16/how-much-bias-is-introduced-by-the-remainder-technique/
int RandomNumberGenerator::random_integer_uniform(int min, int max)
{

    //LOG->write("min: ", false);
    //LOG->write(min);
    //LOG->write("max: ", false);
    //LOG->write(max);

    if (max == 0)
    {
        //LOG->write("MAX WAS 0");
        int range = min;
        while (true)
        {
            int value = xorshf96();
            if (value < RAND_MAX - RAND_MAX % range)
            {

                //LOG->write("range: ", false);
                //LOG->write(range);
                //LOG->write("value mod range: ", false);
                //LOG->write(value % range);

                return value % range;
            }
        }
    }
    else
    {
        assert(max > min);
        int range = max - min;
        while (true)
        {
            int value = xorshf96();
            if (value < RAND_MAX - RAND_MAX % range)
            {   
                int res = min + (value % range);
                assert(res < max);
                assert(res >= min);
                return res;
            }
        }
    }
}

// chooses a random integer from {0,1,2, range_max - 1}
int RandomNumberGenerator::random_integer_uniform(int range_max)
{
    return random_integer_uniform(0, range_max);
}

double RandomNumberGenerator::random_0_1_double()
{   
    // cout << endl << xorshf96() << endl;
    // cout << (double) xorshf96() / (double) INT_MAX << endl;
    return (double) xorshf96() / (double) INT_MAX;
}

void GenerateRandomPermutation(int *permutation, int n)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in GenerateRandomPermutation." << endl;
    #endif
    RandomNumberGenerator* rng = new RandomNumberGenerator();
    rng->seed();
    GenerateRandomPermutation(permutation, n, rng);
    delete rng;
    
}

void GenerateRandomPermutation(int *permutation, int n, RandomNumberGenerator* rng)
{
    for (int i = 0; i < n; ++i)
    {
        permutation[i] = i;
    }
    shuffle_vector(permutation, n, rng);
    assert(isPermutation(permutation, n)) ;

}


void GenerateRandomRealvec_0_1(double *real_vec, int n)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in GenerateRandomPermutation." << endl;
    #endif
    RandomNumberGenerator* rng = new RandomNumberGenerator();
    rng->seed();
    GenerateRandomRealvec_0_1(real_vec, n, rng);
    delete rng;
}

void GenerateRandomRealvec_0_1(double *real_vec, int n, RandomNumberGenerator* rng)
{
    for (int i = 0; i < n; ++i)
    {
        real_vec[i] = rng->random_0_1_double();
    }
}


int count_n_dif_array_items_double(double* array1, double* array2, int n){
    int res = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(array1[i]- array2[i]) > SMALLEST_POSITIVE_DOUBLE)
        {
            res++;
        }
    }
    return res;
}

int count_n_dif_matrix_items_double(double** matrix1, double** matrix2, int n, int m){
    int res = 0;
    for (int i = 0; i < m; i++)
    {
        res += count_n_dif_array_items_double(matrix1[i], matrix2[i], n);
    }
    return res;
}






double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

int choose_index_given_probabilities(double *probabilities_array, int len)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in choose_index_given_probabilities." << endl;
    #endif
    RandomNumberGenerator* tmp_rng = new RandomNumberGenerator();
    tmp_rng->seed();
    int res = choose_index_given_probabilities(probabilities_array, len, tmp_rng);
    delete tmp_rng;
    return res;
}

int choose_index_given_probabilities(double *probabilities_array, int len, RandomNumberGenerator* rng)
{   
    
    double r = rng->random_0_1_double();
    double cum_prob = 0;

    for (int i = 0; i < len; i++)
    {
        cum_prob += probabilities_array[i];
        if (r < cum_prob)
        {
            return i;
        }
    }

    return choose_index_given_probabilities(probabilities_array, len, rng);

    //cout << endl;
    //cout << "cum_prob = " << cum_prob << endl;
    //assert(cum_prob > 0.99999);
}

int choose_index_given_weights(double *weights_array, int len){
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in choose_index_given_weights." << endl;
    #endif
    RandomNumberGenerator* rng = new RandomNumberGenerator;
    rng->seed();
    int res = choose_index_given_weights(weights_array, len, rng);
    delete rng;
    return res;
}


int choose_index_given_weights(double *weights_array, int len, RandomNumberGenerator* rng)
{
    double r = rng->random_0_1_double();
    double cum_sum = 0;
    double total = 0;

    for (int i = 0; i < len; i++)
    {
        total += weights_array[i];
    }

    r = total * r;

    for (int i = 0; i < len; i++)
    {
        cum_sum += weights_array[i];
        if (r < cum_sum)
        {
            return i;
        }
    }

    return choose_index_given_weights(weights_array, len, rng);

    //cout << endl;
    //cout << "cum_prob = " << cum_prob << endl;
    //assert(cum_prob > 0.99999);
}

bool coin_toss(double p_of_true)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in coin_toss." << endl;
    #endif
    RandomNumberGenerator* rng = new RandomNumberGenerator();
    rng->seed();
    bool res = coin_toss(p_of_true, rng);
    delete rng;
    return res;
}


bool coin_toss(double p_of_true, RandomNumberGenerator* rng)
{
    if (rng->random_0_1_double() < p_of_true)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int tools_round(double x)
{
    if (x <= 0.0)
    {
        return (int)(x - 0.5);
    }
    else
    {
        return (int)(x + 0.5);
    }
}


void shuffle_vector(int *vec, int len)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in shuffle_vector." << endl;
    #endif
    RandomNumberGenerator* rng = new RandomNumberGenerator();
    rng->seed();
    shuffle_vector(vec, len, rng);
    delete rng;
}



void shuffle_vector(int *vec, int len, RandomNumberGenerator* rng)
{
    for (int i = 0; i < len - 1; i++)
    {
        int pos = rng->random_integer_fast(i, len);
        //int pos = (int) (unif_rand() * (len-i) + i);
        int aux = vec[i];
        vec[i] = vec[pos];
        vec[pos] = aux;
    }
}


namespace PERMU
{

PermuTools::PermuTools(int n)
{
    #ifndef NDEBUG
    std::cout << "WARNING: new rng created in PermuTools creation." << endl;
    #endif
    this->rng = new RandomNumberGenerator;
    rng->seed();
    this->delete_rng_is_required = true;
    init_class(n);
}

PermuTools::PermuTools(int n, RandomNumberGenerator* rng)
{
    this->rng = rng;
    init_class(n);
}

void PermuTools::init_class(int n){
    this->n = n;
    linear_assigment_problem = new Lap(n);

    lap_rows=new int[n];
    lap_cols=new int[n]; 
    lap_u=new int[n];
    lap_v=new int[n];



    random_permu1 = new int[n];
    random_permu2 = new int[n];
    temp_array = new int[n];
    temp_array2 = new int[n];
    temp_array3 = new int[n];
    temp_array4 = new int[n];
    temp_array_double = new double[TEMP_double_ARRAY_SIZE];

    identity_permu = new int[n];
    first_marginal = new double*[n];
    order_marginal = new double*[n];
    freq_matrix = new int*[n];

    hamming_mm_consensus = new int[n];
    kendall_mm_consensus = new int[n];


    for (int i = 0; i < n; i++)
    {
        first_marginal[i] = new double[n];
        order_marginal[i] = new double[n];
        freq_matrix[i] = new int[n];
    }

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            freq_matrix[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++)
    {
        identity_permu[i] = i;
    }
    
    GenerateRandomPermutation(random_permu1, n, rng);
    GenerateRandomPermutation(random_permu2, n, rng);
    GenerateRandomPermutation(temp_array, n, rng);
}




PermuTools::~PermuTools()
{
    if (this->delete_rng_is_required)
    {
        delete rng;
    }

    delete linear_assigment_problem;

    delete[] this->lap_rows; 
    delete[] this->lap_cols; 
    delete[] this->lap_u; 
    delete[] this->lap_v; 


    delete[] this->random_permu1;
    delete[] this->random_permu2;
    delete[] this->temp_array;
    delete[] this->temp_array2;
    delete[] this->temp_array3;
    delete[] this->temp_array4;
    delete[] this->hamming_mm_consensus;
    delete[] this->kendall_mm_consensus;
    delete[] this->temp_array_double;
    delete[] this->identity_permu;

    for (int i = 0; i < n; i++)
    {
        delete[] order_marginal[i];
        delete[] first_marginal[i];
        delete[] freq_matrix[i];
    }
    delete[] order_marginal;
    delete[] first_marginal;
    delete[] freq_matrix;
}


/* 
// combines the permutations considering the coefficients simmilarly to \cite{wang_discrete_2012}. 
// The zeroes on their paper are -1 in our implementation
void PermuTools::combine_permus(int** permu_list, double* coef_list, int* res){
    int m = NEAT::N_PERMU_REFS;
    int non_zero = 0; // number of non-zero coef.
    int positive = 0; // number of strictly positive coef
    int zero = 0; //number of zero coef

    for (int i = 0; i < m; i++)
    {
         if(coef_list[i] < 0)
        {
            non_zero++;
        }else if (coef_list[i] > 0)
        {
            non_zero++;
            positive++;
        }else{
            zero++;
        }
    }

    if (positive == 0)
    {
        GenerateRandomPermutation(res, n);
    }

    QuickSort2Desc(coef_list, permu_list, 0, m - 1, false);

    // normalize positive weights
    double sum_of_pos_w = sum_slice_vec(coef_list, 0, positive);
    double *coef_list_copy = new double[NEAT::N_PERMU_REFS];

    std::copy(coef_list, coef_list+NEAT::N_PERMU_REFS, coef_list_copy);

    //honarte ondo
    for (int i = 0; i < positive; i++)
    {
        coef_list_copy[i] /= sum_of_pos_w;
    }
    
    // normalize_neg_weights, considering their relative weight with respect to pos weights
    double sum_of_neg_w = -sum_slice_vec(coef_list_copy,positive,m);
    for (int i = positive; i < m; i++)
    {
        coef_list_copy[i] /= -(sum_of_pos_w + sum_of_neg_w);
    }

    for (int i = 0; i < n; i++)
    {
        int idx = choose_index_given_probabilities(coef_list_copy, positive);
        res[i] = permu_list[idx][i];
    }



    for (int i = positive + zero; i < m; i++)
    {
        double r = 0;
        for (int j = 0; j < n; j++)
        {
            r = random_0_1_double();
            if (r < coef_list_copy[i])
            {
                if (res[j] == permu_list[i][j])
                {
                    res[j] = -1;
                }
            }
        }
    }
    convert_to_permu(res);
    delete[] coef_list_copy;
}  
*/

/* // Auxiliary function for combine_permus 
void PermuTools::convert_to_permu(int* res){
    std::set<int> existing;
    std::set<int> missing(this->identity_permu, this->identity_permu+n);
    
     // temp array contains the empty positions
    for (int i = 0; i < n; i++)
    {
        temp_array[i] = -1;
    }
    shuffle_vector(random_permu1, n);
    for (int i = 0; i < n; i++)
    {
        int r = random_permu1[i];
        if (res[r] == -1)
        {
            temp_array[r] = r;
        } else if (!existing.insert(res[r]).second) // if not succesfully inserted means it already was in existing.
        {
            temp_array[r] = r;
        }else{
            missing.erase(res[r]);
        }
    }
    shuffle_vector(temp_array, n);
    for (int i = 0; i < n; i++)
    {
        if (temp_array[i] == -1){
            continue;
        }
        int el = *missing.begin();
        missing.erase(missing.begin());
        res[temp_array[i]] = el;
    }
}
*/


void PermuTools::compute_first_marginal(int** permu_list, int m){
    // in the article "Exploiting Probabilistic Independence for Permutations", they define the first marg in the
    // order used in this implementation.

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            this->first_marginal[i][j] = 0;
        }
    }

    double normalized_base_freq = 1.0 / ((double) m * (double) n);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            first_marginal[permu_list[i][j]][j] += normalized_base_freq;
        }
    }
}


void PermuTools::compute_order_marginal(int** permu_list, int m){
    // in the article "Exploiting Probabilistic Independence for Permutations", they define the first marg in the
    // order used in this implementation.

    if (it_to_compute_order_marginal == 0){
        it_to_compute_order_marginal = COMPUTE_ORDER_MARGINAL_EVERY_K_ITERATIONS;
    }else{
        it_to_compute_order_marginal--;
        return;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            this->order_marginal[i][j] = 0;
        }
    }

    double normalized_base_freq = 1.0 / ((double) (n*(DEPTH_OF_ORDER_MARGINAL)/2) * (double) m);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n - DEPTH_OF_ORDER_MARGINAL; j++)
        {
            for (int k = j+1; k < j+ DEPTH_OF_ORDER_MARGINAL; k++)
            {
                if (permu_list[i][j] > permu_list[i][k]) // sum 1 to pos jk only if the number j is before number k
                {
                    order_marginal[permu_list[i][j]][permu_list[i][k]] += normalized_base_freq;
                }
            }
        }
    }
}
#undef COMPUTE_ORDER_MARGINAL_EVERY_K_ITERATIONS




double PermuTools::get_distance_to_order_marginal(int* permu){
    double res = 0.0;

    for (int j = 0; j < n; j++)
    {
        for (int k = j+1; k < min(n, j+ DEPTH_OF_ORDER_MARGINAL); k++)
        {
            if (permu[j] > permu[k])
            {
                res += order_marginal[permu[j]][permu[k]];
            }
        }
    }
    return res;
}
#undef DEPTH_OF_ORDER_MARGINAL

double PermuTools::get_distance_to_marginal(int* permu){
    double res = 0.0;
    for (int i = 0; i < n; i++)
    {
        res += this->first_marginal[permu[i]][i];
    }
    return res;
}

int PermuTools::choose_permu_index_to_move(double* coef_list){
    return choose_permu_index_to_move(coef_list, this->rng);
}


int PermuTools::choose_permu_index_to_move(double* coef_list, RandomNumberGenerator* input_rng){

    assert(TEMP_double_ARRAY_SIZE >= PERMU::N_PERMU_REFS);
    
    for (int i = 0; i < PERMU::N_PERMU_REFS; i++)
    {
        temp_array_double[i] = abs(coef_list[i]);
    }

    if(sum_abs_val_slice_vec(temp_array_double, 0, PERMU::N_PERMU_REFS) == 0.0){
        return -1;
    }

    return choose_index_given_weights(temp_array_double, PERMU::N_PERMU_REFS, input_rng);
}



double PermuTools::compute_kendall_distance(int* permu_1, int* permu_2){
    return (double) Kendall(permu_1, permu_2, n, temp_array, temp_array2, temp_array3, temp_array4);
}

void PermuTools::compute_kendall_consensus_borda(int **permu_list, int m)
{

    for (int i = 0; i < n; i++)
    {
        temp_array[i] = 0;
    }
    
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            temp_array[j] += permu_list[i][j];
        }
    }

    compute_order_from_int_to_int(temp_array, n, kendall_mm_consensus);

}

void PermuTools::compute_hamming_consensus(int **permu_list, int m)
{


    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            freq_matrix[j][permu_list[i][j]]--;

    linear_assigment_problem->seedRandom(rng->x);
    linear_assigment_problem->execute_lap(freq_matrix, lap_rows, lap_cols, lap_u, lap_v);

    for (int i = 0; i < n; i++)
        hamming_mm_consensus[i] = lap_rows[i];

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            freq_matrix[j][permu_list[i][j]] = 0;

}

double PermuTools::compute_hamming_distance(int* permu_1, int* permu_2){
    return (double) Hamming_distance(permu_1, permu_2, n);
}


}




// https://thispointer.com/c-how-to-read-a-file-line-by-line-into-a-vector/
std::vector<string> read_lines_from_file(string filename){
    std::ifstream in(filename);
    std::string str;
    std::vector<std::string> vecOfStrs;

    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (str.size() > 0)
        {
            vecOfStrs.push_back(str);
        }
    }
    in.close();
    if (vecOfStrs.size()< 1){
        std::cout << endl;
        std::cout << "file: \"" << filename << "\" not read correctly." << endl;
        exit(1);
    }
    return vecOfStrs;
}





std::vector<string> split(string txt, char ch)
{
    size_t pos = txt.find(ch);
    size_t initialPos = 0;
    std::vector<std::string> res;

    // Decompose statement
    while( pos != std::string::npos ) {
        res.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    res.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return res;
}



// if NOT reverse, then the smallest value (0.0 in the case of the normalized result) will NOT change position
void transform_from_values_to_normalized_rankings(double* reference_and_result, int len, bool reverse){
    double* res = new double[len];
    
    assert(len >2);
    compute_order_from_double_to_double(reference_and_result, len, res, reverse);
    
    for (int i = 0; i < len; i++)
    {
        reference_and_result[i] = res[i] / (double) (len - 1);
    }

    delete[] res;
}






void compute_order_from_int_to_int(int* v, int len, int* order_res, bool reverse){
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }


    int* temp = new int[len];
    for (int i = 0; i < len; i++)
    {
        temp[i] = i;
    }
    std::sort(temp, temp+len, _sort_indices(v));
    for (int i = 0; i < len; i++)
    {
        order_res[temp[i]] = i;
    }
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }
    delete[] temp;
}

void compute_order_from_double_to_int(double* v, int len, int* order_res, bool reverse){
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }


    int* temp = new int[len];
    for (int i = 0; i < len; i++)
    {
        temp[i] = i;
    }
    std::sort(temp, temp+len, _sort_indices_double(v));
    for (int i = 0; i < len; i++)
    {
        order_res[temp[i]] = i;
    }
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }
    delete[] temp;
}


bool are_doubles_equal(double x1, double x2)
{
    if (fabs(x1 - x2) < SMALLEST_POSITIVE_DOUBLE)
    {
        return true;
    }else{
        return false;
    }
    
}


void compute_order_from_double_to_double(double* v, int len, double* order_res, bool reverse, bool respect_ties){
    int* temp = new int[len];

    double* copy_of_v = new double[len];
    memcpy(copy_of_v, v, sizeof(double)*len);

    // PrintArray(v, len);
    if (reverse)
    {
        multiply_array_with_value(v, -1, len);
    }
    
    for (int i = 0; i < len; i++)
    {
        temp[i] = i;
    }

    std::sort(temp, temp+len, _sort_indices_double(v));

    if (reverse)
    {
        multiply_array_with_value(v, -1, len);
    }

    for (int i = 0; i < len; i++)
    {
        order_res[temp[i]] = (double) i;
    }


    
    
    if (respect_ties)
    {
        if (len == 1)
        {
            v[0] = 0.0;
            return;
        }
        assert(len  >= 2);
        int n_equal = 0;
        double last;


        for (int i = 1; i < len; i++)
        {
            last = copy_of_v[temp[i-1]];

            if (are_doubles_equal(last, copy_of_v[temp[i]]))
            {
                n_equal++;
            }

            if (!are_doubles_equal(last, copy_of_v[temp[i]]) || i == len - 1)
            {
                int last_eq_idx = i-1;

                if(are_doubles_equal(last, copy_of_v[temp[i]])){
                    last_eq_idx = i;
                }

                if (n_equal > 0)
                {
                    double av_rank = (order_res[temp[last_eq_idx]] + order_res[temp[last_eq_idx-n_equal]]) / 2.0;

                    for (int j = 0; j <= n_equal ; j++)
                    {
                        order_res[temp[last_eq_idx-j]] = av_rank;
                    }
                    n_equal = 0;
                }
            }
        }
    }

    // PrintArray(order_res, len);
    // cout << endl;
    // cout << "---\n";
        
    delete[] copy_of_v;
    delete[] temp;
}










// 1-p will be assigned to the best individual, (1-p)*p to the second one etc.
void transform_from_values_to_geometric_ranking_probs(double* reference_and_result, int len, bool reverse){

    int p = 0.8;

    int* indexes = new int[len];
    
    assert(len >2); 

    compute_order_from_double_to_int(reference_and_result, len, indexes, reverse);
    reference_and_result[indexes[0]] = 1.0 - p;
    for (int i = 1; i < len; i++)
    {
        reference_and_result[indexes[i]] = reference_and_result[indexes[i-1]] * p;
    }

    normalize_vector(reference_and_result, len);



    delete[] indexes;
}
#undef p

double tools_round_two_decimals(double x)
{

     // 37.66666 * 100 =3766.66 
    // 3766.66 + .5 =3767.16    for rounding off value 
    // then type cast to int so value is 3767 
    // then divided by 100 so the value converted into 37.67 
    float value = (int)(x * 100 + .5); 
    return (float)value / 100; 

}



double obtain_variance_considering_repetitions(double length, double* array_of_values){
    double n = length;
    double N = length * 2.0;


    double res = n * n / 12.0;

    double sumand = 0.0;

    // compute sumand
    {

        std::vector<int> group_indexes;



        bool* not_yet_considered = new bool[tools_round(length*2.0)];

        for (int i = 0; i < tools_round(length*2.0); i++)
        {
            not_yet_considered[i] = true;
        }




        for (int i = 0; i < tools_round(length*2.0); i++)
        {
            if(not_yet_considered[i])
            {
                double val = array_of_values[i];
                int n_of_group_members = 0;
                for (int j = i; j < tools_round(length*2.0); j++)
                {
                    if (not_yet_considered[j] && are_doubles_equal(val, array_of_values[j]))
                    {
                        group_indexes.push_back(j);
                        n_of_group_members++;
                    }
                }

                sumand += (double) (n_of_group_members*n_of_group_members*n_of_group_members - n_of_group_members);

                for (auto &index : group_indexes)
                {
                    not_yet_considered[index] = false;
                }

                group_indexes.clear();
                n_of_group_members = 0;
            }
        }


        delete[] not_yet_considered;

    }

    res *= (N + 1.0) - (sumand / (N * N-1));

    return sqrt(res);

}



double from_u_statistic_to_z(double u, double length, double* array_of_values){

    double n = (double) length;
    double N = (double) length * 2.0;

    double mu = (n * n) / 2.0;

    double sigma = sqrt( n * n * (N + 1.0) / 12.0 );

    double sigma_corrected = obtain_variance_considering_repetitions(length, array_of_values);

    //cout << "--- sigma -> " << sigma << ",   sigma_corrected -> " << sigma_corrected << endl;

    double z = (u - mu) / sigma;

    return z;
}



void load_statistical_sigificant_parameters_given_alpha_index(int ALPHA_INDEX, double &ALPHA, double &Z_THRESH, int *critical_values_one_sided = nullptr)
{
    
    if (ALPHA_INDEX == 0)
    {
        ALPHA = 0.05;
        Z_THRESH = 1.645;
        if (critical_values_one_sided != nullptr)
        {
            int crit_vals[10] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 15,      17,      22,      26, 29};
            copy_array(critical_values_one_sided, crit_vals, 10);
        }
    } 
    else if(ALPHA_INDEX == 1)
    {
        ALPHA = 0.01;
        Z_THRESH = 2.326;
        if (critical_values_one_sided != nullptr)
        {
            int crit_vals[10] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 28,      34, 39};
            copy_array(critical_values_one_sided, crit_vals, 10);
        }
    }
    else if (ALPHA_INDEX == 2)
    {
        ALPHA = 0.005;
        Z_THRESH = 2.576;
        if (critical_values_one_sided != nullptr)
        {
            int crit_vals[10] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 36, 43};
            copy_array(critical_values_one_sided, crit_vals, 10);
        }
    }
    else
    {
        cout << "ERROR:  Only indexes 0, 1 and 2 allowed in alpha selection.";
    }
}




//Unpaired test
bool is_A_larger_than_B_Mann_Whitney(double* A, double* B, int length, int ALPHA_INDEX){

    double Z_THRESH;
    double ALPHA;

    load_statistical_sigificant_parameters_given_alpha_index(ALPHA_INDEX, ALPHA, Z_THRESH);

    if(length < 20){
        cout << "A larger sample size than 20 is required to correctly estimate p-value. " << endl;
        exit(1);
    }

    double *array_of_values = new double[length*2];
    double *ranks = new double[length*2];

    for (int i = 0; i < length; i++)
    {
        array_of_values[i] = A[i];
        array_of_values[i + length] = B[i];
    }

    compute_order_from_double_to_double(array_of_values, length*2, ranks, false, true);
    sum_value_to_array(ranks, 1.0, length*2);

    double r_A = sum_slice_vec(ranks, 0, length);
    double r_B = sum_slice_vec(ranks, length, length*2);



    if(r_A < r_B)
    {
        //cout << "One sided test not performed, z < 0 was found" << endl;
        return false;
    }




    //double u_A = length * length + (length * (length+ 1) / 2) - r_A;
    double u_B = length * length + (length * (length+ 1) / 2) - r_B;

    double z = from_u_statistic_to_z(u_B, length, array_of_values);

    
    delete[] array_of_values;
    delete[] ranks;

    //cout << "Unpaired rank hypotesis testing at alpha =  "<< ALPHA <<  endl;

    //cout << "z = " << z << " ";

    if (z > Z_THRESH)
    {   
        //cout << " > z_thresh = " << Z_THRESH << ", critical diff. found." << endl;
        return true;
    }else{
        //cout << " < z_thresh = " << Z_THRESH << ", no crit. diff." << endl;
        return false;
    }

}


// Paired test 
// http://vassarstats.net/textbook/ch12a.html
//    example:
//
//    vector<double> A{78, 24, 64, 45, 64, 52, 30, 50, 64, 50, 78, 22, 84, 40, 90, 72};
//    vector<double> B{78, 24, 62, 48, 68, 56, 25, 44, 56, 40, 68, 36, 68, 20, 58, 32};
//    cout << "\n\n" << is_A_larger_than_B_Signed_Wilcoxon(A.data(), B.data(), 16) << endl;
//    exit(0);
bool is_A_larger_than_B_Signed_Wilcoxon(double* A, double* B, int length, int ALPHA_INDEX){


    if(length < 4){
        cout << "A larger sample size than 4 is required to correctly estimate p-value. " << endl;
        exit(1);
    }

    //cout << "length: " << length << endl; 

    // double *signs_0s_discarded = new double[length];
    // double *abs_differences_0s_discarded = new double[length];
    vector<double> ranks;
    vector<int> signs;
    vector<double> abs_differences;

    for (int i = 0; i < length; i++)
    {
        double abs_diff = abs(A[i] - B[i]);
        int sign_diff = sign(A[i] - B[i]);
        if (sign_diff != 0)
        {
            signs.push_back(sign_diff);
            abs_differences.push_back(abs_diff);
        }
    }

    int N_r = abs_differences.size();
    ranks.resize(N_r);

    if (N_r == 0)
    {
        // Samples are equal
        return false;
    }
    

    compute_order_from_double_to_double(abs_differences.data(), N_r, ranks.data(), false, true);
    sum_value_to_array(ranks.data(), 1.0, N_r);

    double W = scalar_multiplication(ranks.data(), signs.data(), N_r);


    // cout << "----" << endl;
    // print_vector(ranks);
    // cout << endl;
    // print_vector(signs);
    // cout << endl;

    //cout << "W: " << W << endl;

    if(W < 0) // one sided test, W must be positive.
    {
        //cout << "One sided test not performed, z < 0 was found" << endl;
        return false;
    }

    double ALPHA;
    double Z_THRESH;
    int critical_values_one_sided[10] = {0};

    load_statistical_sigificant_parameters_given_alpha_index(ALPHA_INDEX, ALPHA, Z_THRESH, critical_values_one_sided);

    double d_N_r = (double) N_r;
 
    double sigma_w = sqrt(d_N_r * (d_N_r + 1.0) * (2.0*d_N_r + 1.0) / 6.0);

    //cout << "sigma_w: " << sigma_w << endl;

    double z = W / sigma_w;

    if (N_r <= 9)
    {
        if (W > critical_values_one_sided[N_r])
        {
            return true;
        }else
        {
            return false;
        }
    }
    

    //cout << "Paired rank hypothesis testing at alpha =  "<< ALPHA <<  endl;

    //cout << "z = " << z << " ";

    if (z > Z_THRESH)
    {   
        //cout << " > z_thresh = " << Z_THRESH << ", critical diff. found." << endl;
        return true;
    }else{
        //cout << " < z_thresh = " << Z_THRESH << ", no crit. diff." << endl;
        return false;
    }
}




double p_value_chisquared(double x, double df)
{
   cout << "Function removed to avoid using other people's code and thus increase License burdens." << endl;
   exit(1);
}


void get_ranks_from_f_values(vector<vector<double>>& ranks, double** f_values, int n_candidates, int n_samples)
{
    static vector<double> scores;
    for (int i = 0; i < n_samples; i++)
    {   
        scores.clear();
        for (int j = 0; j < n_candidates; j++)
        {
            scores.push_back(f_values[j][i]);
        }
        // conover practical statistics page 381 -> rank 1 assigned to the lowest value
        compute_order_from_double_to_double(scores.data(), n_candidates, scores.data(), false, true);
        sum_value_to_array(scores.data(), 1.0, n_candidates);
        for (int j = 0; j < n_candidates; j++)
        {
            ranks[j][i] = scores[j];
        }
    }
}


bool Friedman_test_are_there_critical_diferences(double** f_values, int n_candidates, int n_samples, int ALPHA_INDEX)
{
    // value_1 of controller_1, value_2 of controller_1, ..., value_sample_length of controller_1
    // value_1 of controller_2, value_2 of controller_2, ..., value_sample_length of controller_2
    // ...
    // value_1 of controller_n_candidates, value_2 of controller_n_candidates, ..., value_sample_length of controller_n_candidates

    double ALPHA;
    double Z_THRESH;

    load_statistical_sigificant_parameters_given_alpha_index(ALPHA_INDEX, ALPHA, Z_THRESH);

    static vector<vector<double>> ranks;

    ranks.resize(n_candidates);
    for (int i = 0; i < ranks.size(); i++)
    {
        ranks[i].resize(n_samples);
    }

    int k = n_samples;
    int m = n_candidates;


    get_ranks_from_f_values(ranks, f_values, n_candidates, n_samples);


    double numerator = 0.0;

    for (int j = 0; j < m; j++)
    {
        double sumand = (sum_slice_vec(ranks[j].data(), 0, n_samples) - ((double)k * ((double)m + 1.0) * 0.5));
        sumand *= sumand;
        numerator += sumand;
    }

    numerator *= (double) (m - 1);
    
    double denominator = 0.0;
    for (int l = 0; l < k; l++)
    {
        for (int j = 0; j < m; j++)
        {
            denominator += ranks[j][l] * ranks[j][l];
        }
    }
    
    // for (int j = 0; j < n_candidates; j++)
    // {
    //     print_vector(ranks[j]);
    //     cout << endl;
    // }



    denominator -= (double)(k*m*(m+1)*(m+1)) / 4.0;

    // cout << numerator << " / " << denominator << endl;


    double T = numerator / denominator;

    // cout << "T = " << T << endl;

    if(T < 0)
    {
        cout << "T is negative\n" << std::flush;
    }

    int df = m - 1;
    double p = p_value_chisquared(T, (double) df);

    


    if (p > ALPHA)
    {
        cout << "Friedman H0 ";
        cout << "T: " << T << ", df: " << df << ", Friedman p-value: " << p << std::flush;
        return false;
    }
    else
    {
        cout << "Friedman H1 ";
        cout << "T: " << T << ", df: " << df << ", Friedman p-value: " << p << std::flush;
        return true;
    }
}


void F_race_iteration(double** f_values, vector<int> &surviving_candidates, int n_samples, int ALPHA_INDEX)
{
    double** reduced_f_values = new double*[surviving_candidates.size()];
    static vector<vector<double>> ranks;

    ranks.resize(surviving_candidates.size());
    for (int i = 0; i < ranks.size(); i++)
    {
        ranks[i].resize(n_samples);
    }
    vector<int> surviving_candidates_next_iteration;
    for (int i = 0; i < surviving_candidates.size(); i++)
    {
        reduced_f_values[i] = f_values[surviving_candidates[i]];
    }

    get_ranks_from_f_values(ranks, reduced_f_values, surviving_candidates.size(), n_samples);

    //PrintMatrix(reduced_f_values, surviving_candidates.size(), n_samples);
    // cout << "\n ---- \n";
    // PrintMatrix(reduced_f_values, surviving_candidates.size(), n_samples);


    if (surviving_candidates.size() == 2 || Friedman_test_are_there_critical_diferences(reduced_f_values, surviving_candidates.size(), n_samples, ALPHA_INDEX))
    {
        int best_reduced_index = 0;
        double* avg_ranks = new double[surviving_candidates.size()];
        for (int i = 0; i < surviving_candidates.size(); i++)
        {
            avg_ranks[i] = Average(ranks[i].data(), n_samples);
        }
        best_reduced_index = argmax(avg_ranks, surviving_candidates.size());

        // cout << endl;
        // cout << "avg_raks: ";
        // PrintArray(avg_ranks, surviving_candidates.size());
        // cout << endl;
        // cout << endl;
        // cout << "best_reduced_index: " << best_reduced_index << endl;
        delete[] avg_ranks;

        for (int i = 0; i < surviving_candidates.size(); i++)
        {
            // PrintArray(reduced_f_values[i], n_samples);
            // cout << endl;
            // PrintArray(reduced_f_values[best_reduced_index], n_samples);
            // cout << endl;
            // cout << "---\n";
            bool pos_hoc_test_result = is_A_larger_than_B_Signed_Wilcoxon(reduced_f_values[best_reduced_index], reduced_f_values[i], n_samples, ALPHA_INDEX);

            // statistically significantly worse than best of generation
            if (pos_hoc_test_result)
            {


            }
            else
            {
                surviving_candidates_next_iteration.push_back(surviving_candidates[i]);
            }
        }
        surviving_candidates = surviving_candidates_next_iteration;
    }
    delete[] reduced_f_values;
}


std::string from_path_to_filename(std::string file_path)
{
    std::string filename = file_path;
    // Remove directory if present.
    // Do this before extension removal incase directory has a period character.
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        filename.erase(0, last_slash_idx + 1);
    }

    // Remove extension if present.
    size_t period_idx = filename.rfind('.');
    while (std::string::npos != period_idx)
    {   
        filename.erase(period_idx);
        period_idx = filename.rfind('.');
    }

    return filename;
}



double euclid_dist(double* array_1, double* array_2, int len) // equiv to l2 distance
{
    double res = 0.0;

    for (int i = 0; i < len; i++)
    {
        res += (array_1 - array_2) * (array_1 - array_2);
    }

    return sqrt(res);

}





progress_bar::progress_bar(int n){
    this->timer = stopwatch();
    this->timer.tic();
    this->max_steps = n;
    this->current_steps = 0;
    std::cout << "[" << std::flush;
}

progress_bar::~progress_bar(){
}

double float_reminder(double x, double divisor){
    double res = x / divisor;
    return res - (double) (int) res;
}

void progress_bar::step(){
    #define NUMBER_OF_PROGRESS_DOTS 15
    
    //int NUMBER_OF_PROGRESS_DOTS = min(MAX_NUMBER_OF_PROGRESS_DOTS, max_steps / 10);

    double last_reminder = float_reminder((double) current_steps-1, (double) NUMBER_OF_PROGRESS_DOTS);
    double current_reminder = float_reminder((double) current_steps, (double) NUMBER_OF_PROGRESS_DOTS);
    double next_reminder = float_reminder((double) current_steps+1, (double) NUMBER_OF_PROGRESS_DOTS);

    if ( (int) (NUMBER_OF_PROGRESS_DOTS * current_steps) / max_steps > (int) (NUMBER_OF_PROGRESS_DOTS * (current_steps-1)) / max_steps )
    {
        std::cout << "." << std::flush;
    }
    this->current_steps++;
}

void progress_bar::end(){
    std::cout << "]" << " " << this->timer.toc() << "(s)" << std::flush;
}

void progress_bar::restart(int n){
    this->timer.tic();
    this->max_steps = n;
    this->current_steps = 0;
    std::cout << "[" << std::flush;
}

// https://en.wikipedia.org/wiki/Binomial_coefficient#Computing_the_value_of_binomial_coefficients
int n_choose_k(int n, int k){

    int res = 1;
    for (int i = 1; i <= k; i++)
    {
        res *= (n + 1 - i);
        res /= i;
    }
    if (res < 0)
    {
        cerr << n << " choose " << k << "was negative -> " << res;
        exit(1);
    }
    
    return res;
}


// split string into vector of strings
vector<string> split_string(string str, string token){
    vector<string>result;
    while(str.size()){
        int index = str.find(token);
        if(index!=string::npos){
            result.push_back(str.substr(0,index));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(str);
        }else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

std::string system_exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

double get_runtime_hipatia(){
    std::string time_string = system_exec("sacct -j  ${SLURM_JOB_ID} --format=ElapsedRaw --parsable | sed -n 2p | cut -d '=' -f 2 | sed 's/|$//'");
    int time_sec = atoi(time_string.c_str());
    return (double) time_sec;
}

void mkdir(const string &path)
{
    if (!exists(path))
    {
        int status = ::mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (0 != status)
        {
            char buf[2048];
            sprintf(buf, "Failed making directory '%s'", path.c_str());
            perror(buf);
            if (!exists(path))
            {
                exit(1);
            }
        }
    }
}

bool exists(const std::string &path) {
    struct stat buffer;
    return (stat (path.c_str(), &buffer) == 0);     
}