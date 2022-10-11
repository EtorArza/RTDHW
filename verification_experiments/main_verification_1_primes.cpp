#include <iostream>
#include <iomanip>
#include <cstdlib>

// List of prime number between given range
unsigned long long generatePrime(unsigned long long end){
     for (unsigned long long i=2; i<end; i++) 
    {
        bool prime=true;
        for (unsigned long long j=2; j*j<=i; j++)
        {
            if (i % j == 0) 
            {
                prime=false;
                break;    
            }
        }   
        if(prime) continue;
    }
    return 0;
}

 
int main(int argc, char *argv[])
{
    std::cout << "Start verification 1." << std::endl;
    generatePrime(10e6);
    std::cout << "End." << std::endl;
}