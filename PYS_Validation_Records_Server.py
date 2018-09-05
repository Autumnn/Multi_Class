import os
import numpy as np
import xgboost
import json
import warnings
import pyswarms.backend as P
from pyswarms.backend.topology import Star
from metrics_list import MetricList
from datetime import datetime
from print_log import PrintLog

warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_Valid_npz"
#PATH = "KEEL_Cross_Folder_npz_S"
DIRS = os.listdir(PATH)         #   Get files in the folder

parameters = {'max_depth': (3, 10),         # int
              'learning_rate': (0.01, 0.3),
              'n_estimators': (50, 1000),       #int
              'gamma': (0, 1.),
              'min_child_weight': (1, 10),
              'max_delta_step': (0, 0.3),
              'subsample': (0.5, 1),
              'colsample_bytree': (0.5, 1)}
para_bounds = np.array(list(parameters.values()), dtype=np.float)
para_keys = list(parameters.keys())

def evaluate(para_value):
    keys = para_keys
    dim = len(keys)
#    print(type(individual))
#    print(individual)
    for i in range(dim):
        (lo, up) = para_bounds[i]
        if para_value[i] < lo:
            sentence = 'increase ' + str(keys[i]) + ' : ' + str(para_value[i]) + ' to lower bound !'
            #print(sentence)
            para_value[i] = lo
        elif para_value[i] > up:
            sentence = 'decrease ' + str(keys[i]) + ' : ' + str(para_value[i]) + ' to upper bound !'
            #print(sentence)
            para_value[i] = up

    BayesOp_Parameters = dict(zip(keys, para_value))
    BayesOp_Parameters['silent'] = True
    BayesOp_Parameters['nthread'] = -1
    BayesOp_Parameters['seed'] = 0
    #    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])

    r = np.load(name)
    Feature_train_list = r['F_tr_l']
    Label_train_list = r['L_tr_l']
    Feature_valid_list = r['F_va_l']
    Label_valid_list = r['L_va_l']
    Num_list = len(Feature_train_list)

    Num_Cross_Folders = Num_list
    ml_record = MetricList(Num_Cross_Folders)
    i = 0
    for j in range(Num_list):
        Feature_train = Feature_train_list[j]
        Label_train = Label_train_list[j]
        Feature_valid = Feature_valid_list[j]
        Label_valid = Label_valid_list[j]
        Label_valid.ravel()

        xgb = xgboost.XGBClassifier(**BayesOp_Parameters)
        xgb.fit(Feature_train, Label_train.ravel())
        Label_predict = xgb.predict(Feature_valid)
        ml_record.measure(i, Label_valid, Label_predict, 'weighted')
        i += 1

    result = 1 - ml_record.mean_G()

    return result


def evolution_search(f, para_b):
    begin_time = datetime.now()
    Timestamps_list = []
    Target_list = []
    Parameters_list = []
    keys = list(para_b.keys())
    dim = len(keys)
    plog = PrintLog(keys)

    min = np.ones(dim)
    max = np.ones(dim)
    value_list = list(parameters.values())
    for i_v in range(dim):
        min[i_v] = value_list[i_v][0]
        max[i_v] = value_list[i_v][1]
    bounds = (min, max)
    plog.print_header(initialization=True)

    my_topology = Star()
    my_options ={'c1': 0.6, 'c2': 0.3, 'w': 0.4}
    my_swarm = P.create_swarm(n_particles=20, dimensions=dim, options=my_options, bounds=bounds)  # The Swarm Class

    iterations = 30  # Set 100 iterations
    for i in range(iterations):
        # Part 1: Update personal best

        # for evaluated_result in map(evaluate, my_swarm.position):
        #     my_swarm.current_cost = np.append(evaluated_result)
        # for best_personal_result in map(evaluate, my_swarm.pbest_pos):  # Compute personal best pos
        #     my_swarm.pbest_cost = np.append(my_swarm.pbest_cost, best_personal_result)
        my_swarm.current_cost = np.array(list(map(evaluate, my_swarm.position)))
        #print(my_swarm.current_cost)
        my_swarm.pbest_cost = np.array(list(map(evaluate, my_swarm.pbest_pos)))
        my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)  # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

        # Let's print our output
        #if i % 2 == 0:
        #    print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i + 1, my_swarm.best_cost))

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        my_swarm.velocity = my_topology.compute_velocity(my_swarm)
        my_swarm.position = my_topology.compute_position(my_swarm)

        Parameters_list.append(my_swarm.best_pos.tolist())
        Target_list.append(1-my_swarm.best_cost)
        elapse_time = (datetime.now() - begin_time).total_seconds()
        Timestamps_list.append(elapse_time)
#        print("The best candidate: ", my_swarm.best_pos)
#        print("The best result: ", res[1])
        plog.print_step(my_swarm.best_pos, 1 - my_swarm.best_cost)
        if i == 0:
            plog.print_header(initialization=False)

    return Timestamps_list, Target_list, Parameters_list

save_path = "KEEL_Cross_Folder_XGBoost_Para"

for i_test in range(1, 9):
    for Dir in DIRS:
        print("Data Set Name: ", Dir)
        dir_path = PATH + "/" + Dir
        files = os.listdir(dir_path)  # Get files in the folder
        list_path = save_path + '/PYS_Timeline_Record_Validation_' + str(i_test) + '/' + Dir
        os.makedirs(list_path)
        para_path = save_path + '/PYS_Validation_' + str(i_test) + '/' + Dir
        os.makedirs(para_path)

        for file in files:
            name = dir_path + '/' + file
            print("Data Set Folder: ", file)

            time_list, target_list, para_list = evolution_search(evaluate, parameters)
            max_val = max(target_list)
            target_index = target_list.index(max_val)
            max_params = para_list[target_index]

            sub_name = file.split(".")[0]

            with open('PYS_Opt_G_Mean_Validation_Records_' + str(i_test) + '.txt', 'a') as w:
                line = sub_name + '\t' + str(max_val) + '\n'
                w.write(line)

            para_file = para_path + '/' + sub_name + '_PYS.json'
            Output_Parameters = dict(zip(para_keys, max_params))
            with open(para_file, 'a') as outfile:
                json.dump(Output_Parameters, outfile, ensure_ascii=False)
                outfile.write('\n')

            list_file = list_path + '/' + sub_name + '_PYS_List.json'
            Output_line = {}
            for i in range(len(target_list)):
                Output_line[time_list[i]] = {target_list[i]: para_list[i]}
            with open(list_file, 'a') as outfile:
                json.dump(Output_line, outfile, ensure_ascii=False)
                outfile.write('\n')



