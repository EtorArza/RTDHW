#include "Parameters.h"
#include "Tools.h"
#include "INIReader.h"
#include <cfloat>
#include <stdio.h>
#include "constants.h"

// MACROS //

///////////

neat_parameters::neat_parameters(){};
neat_parameters::~neat_parameters(){};

void neat_parameters::load_global_params(std::string conf_file_path)
{
    INIReader reader(conf_file_path);
    N_OF_THREADS = reader.GetInteger("Global", "THREADS", -1);
    MAX_TRAIN_TIME = reader.GetInteger("Global", "MAX_TRAIN_TIME", -1);
    POPSIZE_NEAT = reader.GetInteger("Global", "POPSIZE", -1);
    EXPERIMENT_FOLDER_NAME = reader.Get("Global", "EXPERIMENT_FOLDER_NAME", "UNKNOWN");
    CONTROLLER_NAME_PREFIX = reader.Get("Global", "CONTROLLER_NAME_PREFIX", "UNKNOWN");
    global_timer.tic();
}

void neat_parameters::delete_prev_exp_folder()
{
    if (exists(EXPERIMENT_FOLDER_NAME + "/all_controllers/" + CONTROLLER_NAME_PREFIX + "_gen_0001.controller") || 
        exists(EXPERIMENT_FOLDER_NAME + "/top_controllers/" + CONTROLLER_NAME_PREFIX + "_best.controller"))
    {
        if (DELETE_PREV_EXPERIMENT_FOLDER)
        {
            cout << "deleting..." << endl;
            system(std::string("rm " + EXPERIMENT_FOLDER_NAME + "/all_controllers/" + CONTROLLER_NAME_PREFIX + "_*").c_str());
            system(std::string("rm " + EXPERIMENT_FOLDER_NAME + "/top_controllers/" + CONTROLLER_NAME_PREFIX + "_*").c_str());
        }
        else
        {
            if (exists(EXPERIMENT_FOLDER_NAME + "/all_controllers/" + CONTROLLER_NAME_PREFIX + "_gen_0001.controller"))
            {
                std::cout << "Already exists: " + EXPERIMENT_FOLDER_NAME + "/all_controllers/" + CONTROLLER_NAME_PREFIX + "_gen_0001.controller" << endl;
            }
            if (exists(EXPERIMENT_FOLDER_NAME + "/top_controllers/" + CONTROLLER_NAME_PREFIX + "_best.controller"))
            {
                std::cout << "Already exists: " + EXPERIMENT_FOLDER_NAME + "/top_controllers/" + CONTROLLER_NAME_PREFIX + "_best.controller" << endl;
            }
            
        
            cout << "Move your controller directories or use -f to delete them automatically. If -f is used, all previous experiments with conflicting names will be deleted." << endl;
            exit(1);
        }
    }
    else
    {

    }
    
}