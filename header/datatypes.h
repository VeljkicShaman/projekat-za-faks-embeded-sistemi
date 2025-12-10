#ifndef DATATYPES_H
#define DATATYPES_H

#define SC_INCLUDE_FX
#include <vector>
#include <systemc.h>


using DATA_T = sc_fixed<32, 10, SC_RND, SC_SAT>; 

using PROB_T = sc_fixed<16, 1, SC_RND, SC_SAT>;

using ACC_T = sc_fixed<32, 12, SC_RND, SC_SAT>; 

using MULT_T = ACC_T; 

using Matrix = std::vector<std::vector<DATA_T>>;
using Vector = std::vector<DATA_T>;
using Matrix3D = std::vector<Matrix>;

#endif // DATATYPES_H