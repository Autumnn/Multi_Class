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


class Dead_GA:

    def __init__(self, para_bounds_dic):
        self.keys = list(para_bounds_dic.keys())
        self.bounds = np.array(list(para_bounds_dic.values()), dtype=np.float)
        self.dim = len(keys)
        self.para_data = np.empty((1, self.dim))

        for col, (lower, upper) in enumerate(self.bounds):
            self.para_data.T[col] = np.random.RandomState().uniform(lower, upper)

        self.para_data = np.asarray(self.para_data).ravel()
        self.parameter_value = self.para_data.tolist()

def para_init():



    para_data =


    return parameter_value


parameters = {'max_depth': (5, 10),         # int
              'learning_rate': (0.01, 0.3),
              'n_estimators': (50, 1000),       #int
              'gamma': (0.01, 1.),
              'min_child_weight': (2, 10),
              'max_delta_step': (0, 0.1),
              'subsample': (0.7, 0.8),
              'colsample_bytree': (0.5, 0.99)}

para_bounds = np.array(list(parameters.values()), dtype=np.float)
para_keys = list(parameters.keys())

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", para_init, parameters)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
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
    BayesOp_Parameters['seed'] = 1234
    #    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])
    Num_Cross_Folders = 5
    ml_record = MetricList(Num_Cross_Folders)
    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Feature_train = r['F_tr']
        Label_train = r['L_tr']
        Num_train = Feature_train.shape[0]
        # print(Num_train)
        Feature_test = r['F_te']
        Label_test = r['L_te']
        Label_test.ravel()
        Num_test = Feature_test.shape[0]
        # print(Num_test)

        xgb = xgboost.XGBClassifier(**BayesOp_Parameters)
        xgb.fit(Feature_train, Label_train.ravel())
        Label_predict = xgb.predict(Feature_test)
        ml_record.measure(i, Label_test, Label_predict, 'weighted')
        i += 1

    return ml_record.mean_G(),


toolbox.register("mate", tools.cxTwoPoint) # mate
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # mutate
toolbox.register("select", tools.selTournament, tournsize=3) # select
toolbox.register("evaluate", evaluate)

def Genetic_Algorithm():
    pop = toolbox.population(n=99)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 30
    plog.print_header(initialization=True)

    fitnesses = list(map(toolbox.evaluate, pop))
    print('start')
    ini_para = {}
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ini_para[fit[0]] = ind
        plog.print_step(ind, fit[0])

    plog.reset_timer()
    plog.print_header(initialization=False)

    opt_para = {}
    metric_value_list = {i: list(ini_para.keys())[i] for i in range(0, len(ini_para))}
    for key_value in metric_value_list:
        opt_para[key_value] = ini_para[metric_value_list[key_value]]

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
            min_metric_value = min(metric_value_list, key=metric_value_list.get)
            if fit[0] > metric_value_list[min_metric_value]:
                change = True
                existing_para = []
                for k, v in metric_value_list.items():
                    if fit[0] == v:
                        existing_para.append(k)
                if len(existing_para) > 0:
                    for index in existing_para:
                        if opt_para[index] == ind:
                            change = False
                if change:
                    metric_value_list[min_metric_value] = fit[0]
                    opt_para[min_metric_value] = ind


            plog.print_step(ind, fit[0])
            plog.reset_timer()

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    # return the result:Last individual,The Return of Evaluate function, The best metric values list and their corresponding parameters.
    return best_ind, best_ind.fitness.values, metric_value_list, opt_para

for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder
    plog = PrintLog(para_keys)

    max_params, max_val, selected_metric_list, selected_optimal_parameters = Genetic_Algorithm()
    print('XGBoost:')
    print("best_values", max_val[0])
    print("best_parameters", max_params)

    with open('GA_Opt_G_Mean.txt', 'a') as w:
        line = Dir + '\t' + str(max_val[0]) + '\n'
        w.write(line)

    Output_Parameters = dict(zip(para_keys, max_params))
    with open(Dir + '_GA.json', 'a') as outfile:
        json.dump(Output_Parameters, outfile, ensure_ascii=False)
        outfile.write('\n')

    Parameters_for_Ensemble = {}
    for keys in selected_metric_list:
        para_dict = dict(zip(para_keys, selected_optimal_parameters[keys]))
        Parameters_for_Ensemble[selected_metric_list[keys]] = para_dict
    with open(Dir + '_Esemble.json', 'a') as esm:
        json.dump(Parameters_for_Ensemble, esm, ensure_ascii=False)
        esm.write('\n')