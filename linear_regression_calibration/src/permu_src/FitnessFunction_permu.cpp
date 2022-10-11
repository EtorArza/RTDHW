#include "FitnessFunction_permu.h"
#include <assert.h>
#include "PBP.h"
#include "Population.h"
#include "Tools.h"
#include <cfloat>
#include "QAP.h"
#include "LOP.h"
#include "PFSP.h"
#include "TSP.h"


// #define COUNTER 1
// #define PRINT 1

namespace PERMU{


void GetProblemInfo(std::string problemType, std::string filename, PBP** problem)
{
    if (problemType == "pfsp")
    {
        (*problem) = new PFSP();
    }
    else if (problemType == "tsp")
    {
        (*problem) = new TSP();
    }
    else if (problemType == "qap")
    {
        (*problem) = new QAP();
    }
    else if (problemType == "lop")
    {
        (*problem) = new LOP();
    }
    // else if (problemType == "api")
    //     problem = new API();
    else
    {
         cout << "Wrong problem type was specified. Problem type \"" << problemType << "\" not recognized."  << endl;
         exit(1);
     }

    //Read the instance.
    (*problem)->Read_with_mutex(filename);

}


}
