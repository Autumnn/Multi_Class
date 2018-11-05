import os
import numpy as np
import xgboost
import random
import warnings
import json
from metrics_list import MetricList
from deap import base
from deap import creator
from deap import tools
from print_log import PrintLog


warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_Valid_npz"
#PATH = "KEEL_Cross_Folder_Valid_npz_S"
DIRS = os.listdir(PATH)

def para_init(para_bounds_dic):
    keys = list(para_bounds_dic.keys())
    bounds = np.array(list(para_bounds_dic.values()), dtype=np.float)
    dim = len(keys)

    para_data = np.empty((1, dim))
    for col, (lower, upper) in enumerate(bounds):
        para_data.T[col] = np.random.RandomState().uniform(lower, upper)

    para_data = np.asarray(para_data).ravel()
    parameter_value = para_data.tolist()
#    assert para_data.size == dim
#    parameter_value = dict(zip(keys, para_data))
    return parameter_value


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

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Individual generator
toolbox.register("attr_bool", para_init, parameters)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
# Population initializers
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    keys = list(parameters.keys())
    dim = len(keys)
#    print(type(individual))
#    print(individual)
    assert len(individual) == dim
    for i in range(dim):
        (lo, up) = para_bounds[i]
        if individual[i] < lo:
            sentence = 'increase ' + str(keys[i]) + ' : ' + str(individual[i]) + ' to lower bound !'
            print(sentence)
            individual[i] = lo
        elif individual[i] > up:
            sentence = 'decrease ' + str(keys[i]) + ' : ' + str(individual[i]) + ' to upper bound !'
            print(sentence)
            individual[i] = up

    BayesOp_Parameters = dict(zip(keys, individual))
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

    return ml_record.mean_G(),


toolbox.register("mate", tools.cxTwoPoint) # mate
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # mutate
toolbox.register("select", tools.selTournament, tournsize=3) # select
toolbox.register("evaluate", evaluate)

def Genetic_Algorithm():
    all_res = {'values': [], 'params': [], 'timestamp': []}
    pop = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 20
    plog.print_header(initialization=True)

    fitnesses = list(map(toolbox.evaluate, pop))
    print('start')
    ini_para = {}
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ini_para[fit[0]] = ind
        plog.print_step(ind, fit[0])

    best_ini = tools.selBest(pop, 1)[0]
    all_res['values'].append(best_ini.fitness.values[0])
    all_res['params'].append(best_ini)

    #    plog.reset_timer()
    #    plog.print_header(initialization=False)
    elapse_time = plog.record_time()
    all_res['timestamp'].append(elapse_time)

#    plog.reset_timer()
#    plog.print_header(initialization=False)
    print("  Evaluated %i individuals" % len(pop))
    print("-- Iterative %i times --" % NGEN)

    for g in range(NGEN):
        if g % 2 == 0:
            print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Change map to list,The documentation on the official website is wrong

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            plog.print_step(ind, fit[0])
#            plog.reset_timer()

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        best_gen = tools.selBest(pop, 1)[0]
        all_res['values'].append(best_gen.fitness.values[0])
        all_res['params'].append(best_gen)
        elapse_time = plog.record_time()
        all_res['timestamp'].append(elapse_time)

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_ind.fitness.values, all_res  # return the result:Last individual,The Return of Evaluate function

save_path = "KEEL_Cross_Folder_XGBoost_Para"

for i_test in range(9, 10):
    for Dir in DIRS:
        print("Data Set Name: ", Dir)
        dir_path = PATH + "/" + Dir
        files = os.listdir(dir_path)  # Get files in the folder
        list_path = save_path + '/GA_Timeline_Record_Validation_' + str(i_test) + '/' + Dir
        os.makedirs(list_path)
        para_path = save_path + '/GA_Validation_' + str(i_test) + '/' + Dir
        os.makedirs(para_path)
    #    plog = PrintLog(para_keys)

        for file in files:
            name = dir_path + '/' + file
            print("Data Set Folder: ", file)
            plog = PrintLog(para_keys)

            max_params, max_val, res_records = Genetic_Algorithm()
            print('XGBoost:')
            print("best_values", max_val[0])
            print("best_parameters", max_params)

            sub_name = file.split(".")[0]

            with open('GA_Opt_G_Mean_Validation_Records_' + str(i_test) + '.txt', 'a') as w:
                line = sub_name + '\t' + str(max_val[0]) + '\n'
                w.write(line)

            para_file = para_path + '/' + sub_name + '_GA.json'
            Output_Parameters = dict(zip(para_keys, max_params))
            with open(para_file, 'a') as outfile:
                json.dump(Output_Parameters, outfile, ensure_ascii=False)
                outfile.write('\n')

            time_list = res_records['timestamp']
            target_list = res_records['values']
            para_list = res_records['params']
            list_file = list_path + '/' + sub_name + '_GA_List.json'
            Output_line = {}
            for i in range(len(target_list)):
                Output_line[time_list[i]] = {target_list[i]: para_list[i]}
            with open(list_file, 'a') as outfile:
                json.dump(Output_line, outfile, ensure_ascii=False)
                outfile.write('\n')
