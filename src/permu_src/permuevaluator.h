#pragma once




namespace PERMU{

    enum operator_t{
    SWAP = 0,
    EXCH = 1,
    INSERT = 2,
    N_OPERATORS = 3,
    };

    void operator++(operator_t &c, int );


    enum sensor_t {
        OPT_SWAP = SWAP, // 1 if local optima for swap neighborhood. 0 else.
        OPT_EXCH = EXCH,
        OPT_INSERT = INSERT,
        RELATIVE_POSITION = 3,
        RELATIVE_TIME = 4,
        DISTANCE = 5,
        SPARSITY = 6,
        ORDER_SPARSITY = 7,
        RELATIVE_POPSIZE = 8,
        RELATIVE_TABU_SIZE = 9,
        __sensor_N = 10,
    };

    const int N_PERMU_REFS = 5;
    enum output_t {
        ls_nothing_move = 0,
        O_ID_SWAP = SWAP + 1,
        O_ID_EXCH = EXCH + 1,
        O_ID_INSERT = INSERT + 1,
        accept_or_reject_worse = 4,
        TABU = 5,
        CHANGE_TABU_SIZE = 6,
        REMOVE_OR_CLONE = 7,
        RANDOM_REINITIALIZE = 8,
        c_hamming_consensus = 9,
        c_kendall_consensus = 10,
        c_pers_best = 11,
        c_best_known = 12,
        c_above = 13,
        __output_N = 14,
    };



}




