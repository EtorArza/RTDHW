#include "permuevaluator.h"

namespace PERMU
{



void operator++( PERMU::operator_t &c, int )
{
    int index = (int) c;
    index++;
    index %= PERMU::N_OPERATORS;
    PERMU::operator_t res;
    c = (PERMU::operator_t) index;
}


}