#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;


// https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/


 
// A utility function that returns
// maximum of two integers
int max(int a, int b) { return (a > b) ? a : b; }
 
// Returns the maximum value that
// can be put in a knapsack of capacity W
int knapSack(int W, int wt[], int val[], int n)
{
 
    // Base Case
    if (n == 0 || W == 0)
        return 0;
 
    // If weight of the nth item is more
    // than Knapsack capacity W, then
    // this item cannot be included
    // in the optimal solution
    if (wt[n - 1] > W)
        return knapSack(W, wt, val, n - 1);
 
    // Return the maximum of two cases:
    // (1) nth item included
    // (2) not included
    else
        return max(
            val[n - 1]
                + knapSack(W - wt[n - 1],
                           wt, val, n - 1),
            knapSack(W, wt, val, n - 1));
}



 
int main(int argc, char *argv[])
{
    std::cout << "Start verification 3." << std::endl;
    

    int size = 30; // number of items
    int W = 1200;     // Capacity of knapsack

  
    int val[size];
    int wt[size];

    for (size_t i = 0; i < size; i++)
    {
        val[i] = i / 40 + i / 5 + i % 3 + i % 9 + i % 19;
        wt[i] = i / 47 + i / 7 + i % 5 + i % 13 + i % 7;
    }

    // Function Call
    knapSack(W, wt, val, size);

    std::cout << "End." << std::endl;
}