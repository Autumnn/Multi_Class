#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:fonttian
@file: Overview.py
@time: 2017/10/15
"""

# Types
from deap import base, creator

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# weights 1.0, 求最大值,-1.0 求最小值
# (1.0,-1.0,)求第一个参数的最大值,求第二个参数的最小值
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialization
import random
import operator
from deap import tools

IND_SIZE = 6  # 种群数

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
# 调用randon.random为每一个基因编码编码创建 随机初始值 也就是范围[0,1]
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operators
# difine evaluate function
# Note that a comma is a must
def evaluate(individual):
#    print(sum(individual))
    return sum(individual),


# use tools in deap to creat our application
toolbox.register("mate", tools.cxTwoPoint) # mate:交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1) # mutate : 变异
toolbox.register("select", tools.selTournament, tournsize=3) # select : 选择保留的最佳个体
## Every time, 1) randomly choose 'tournsize' from 'pop'; 2) among the 'tournsize' chosed ones, return the best
toolbox.register("evaluate", evaluate)  # commit our evaluate


# Algorithms
def main():
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 3

    '''
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    '''

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    temp_para = {}
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        temp_para[fit[0]] = ind

    for individual in pop: print(individual.fitness.values, individual)
    opt_para = {}
    print('optimized parameters:')
    key_list = list(temp_para.keys())
    Sorted_keys = {i: key_list[i] for i in range(0, len(temp_para))}
    print(Sorted_keys)
    for key_value in Sorted_keys:
        opt_para[key_value] = temp_para[Sorted_keys[key_value]]
    p_l = dict(sorted(Sorted_keys.items(), key=operator.itemgetter(1)))
    print(p_l)
    for item in p_l: print(Sorted_keys[item], opt_para[item])

    print("  Evaluated %i individuals" % len(pop))  # 这时候，pop的长度还是300呢
    print("-- Iterative %i times --" % NGEN)

    for g in range(NGEN):
        if g % 1 == 0:     #   if g % 10 == 0:
            print("-- Generation %i --" % g)

        print(Sorted_keys)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Change map to list,The documentation on the official website is wrong
        # print('map after select')
        # for o_i in offspring: print(o_i.fitness.values, o_i)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # print('after cross')
        # for o_i in offspring: print(o_i.fitness.values, o_i)

#        print('mutating')
        for mutant in offspring:
            if random.random() < MUTPB:
#                print('Ori',mutant)
                toolbox.mutate(mutant)
                del mutant.fitness.values
#                print('Aft',mutant)
#         print('after mutant')
#         for o_i in offspring: print(o_i.fitness.values, o_i)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#        print('invalid_ind:')
#        for o_i in invalid_ind: print(o_i.fitness.values, o_i)
#        print('offspring:')
#        for o_i in offspring: print(o_i.fitness.values, o_i)
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            min_key = min(Sorted_keys, key=Sorted_keys.get)
            print(min_key)
            if fit[0] > Sorted_keys[min_key]:
                change = True
                potential_key = []
                for k, v in Sorted_keys.items():
                    if fit[0] == v:
                        potential_key.append(k)
                if len(potential_key) > 0:
                    for index in potential_key:
                        if opt_para[index] == ind:
                            change = False
                            print('Do not change!')
                if change:
                    Sorted_keys[min_key] = fit[0]
                    opt_para[min_key] = ind

        print(Sorted_keys)
        p_l = dict(sorted(Sorted_keys.items(), key=operator.itemgetter(1)))
        print(p_l)
        for item in p_l: print(Sorted_keys[item], opt_para[item])

        print('offspring after mutate:')
        for o_i in offspring: print(o_i.fitness.values, o_i)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    return best_ind, best_ind.fitness.values  # return the result:Last individual,The Return of Evaluate function


if __name__ == "__main__":
    # t1 = time.clock()
    best_ind, best_ind.fitness.values = main()
    # print(pop, best_ind, best_ind.fitness.values)
    # print("pop",pop)
    print("best_ind",best_ind)
    print("best_ind.fitness.values",best_ind.fitness.values)

    # t2 = time.clock()

    # print(t2-t1)
