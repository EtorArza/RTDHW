#pragma once

#include <string>
#include "Tools.h"


class neat_parameters
{
public:
    neat_parameters();
    ~neat_parameters();

    void load_global_params(std::string conf_file_path);
    void delete_prev_exp_folder();

    // PARAMS TO BE SPECIFIED IN CONFIG FILE//
    int N_OF_THREADS;
    double MAX_TRAIN_TIME;
    int POPSIZE_NEAT;
    stopwatch global_timer;

    // Global variables //
    double BEST_FITNESS_TRAIN;
    double N_TIMES_BEST_FITNESS_IMPROVED_TRAIN;
    std::string EXPERIMENT_FOLDER_NAME;
    std::string CONTROLLER_NAME_PREFIX;
    bool DELETE_PREV_EXPERIMENT_FOLDER;
    bool IS_LAST_ITERATION;



};