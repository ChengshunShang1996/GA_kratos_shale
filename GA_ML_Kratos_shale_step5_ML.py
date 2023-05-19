import random
from operator import itemgetter
import os
import shutil
import glob
from distutils.dir_util import copy_tree
import time
import csv
import math
from MachineLearning import MachineLearning
 
class Gene:
    """
    This is a class to represent individual(Gene) in GA algorithom
    each object of this class have two attribute: data, size
    """
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])  # length of gene
 
 
class GA:
    """
    This is a class of GA algorithm.
    """
 
    def __init__(self, parameter):
        """
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:
        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]
 
        """
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter
 
        low = self.parameter[4]
        up = self.parameter[5]
 
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        g = 0
        self.g = g
        
        pop = []
        bestindividual_read = []

        pop_file_name = "G" + str(g) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        bestindividual_file_name = "G" + str(g) + "_bestindividual.txt"
        bestindividual_file_path = os.path.join(os.getcwd(),'kratos_results_data', bestindividual_file_name)

        with open(pop_file_path, "r") as f_r:
            for line in f_r:
                geneinfo = []
                values = [float(s) for s in line.split()]
                for cnt in range(0,12):
                    geneinfo.append(values[cnt])
                pop.append({'Gene': Gene(data=geneinfo), 'fitness': 0.0})     
        f_r.close()

        with open(bestindividual_file_path, "r") as f_r:
            for line in f_r:
                geneinfo = []
                values = [float(s) for s in line.split()]
                for cnt in range(0,12):
                    geneinfo.append(values[cnt])
                bestindividual_read.append({'Gene': Gene(data=geneinfo), 'fitness': values[12]})     
        f_r.close()
        
        self.pop = pop
        self.bestindividual = bestindividual_read  # store the best chromosome in the population

        self.aim_strength = parameter[6]
        self.aim_young_modulus = parameter[7]
        self.indiv_data_head_not_written = True
        self.confining_pressure_list = [0, 5e6, 15e6]
        self.texture_angle_list = [0, 45, 90]
    
    def evaluate_in(self, geneinfo, ML_xgb_4, run_ml_4, ML_xgb_5):
        """
        fitness function using ML model
        """
        x1 = geneinfo[0]
        x2 = geneinfo[1]
        x3 = geneinfo[2]
        x4 = geneinfo[3]
        x5 = geneinfo[4]
        x6 = geneinfo[5]
        x7 = geneinfo[6]
        x8 = geneinfo[7]
        x9 = geneinfo[8]
        x10 = geneinfo[9]
        x11 = geneinfo[10]
        x12 = geneinfo[11]

        low = self.parameter[4]
        low.insert(0, 0)
        low.insert(0, 0)
        up = self.parameter[5]
        up.insert(0, 90)
        up.insert(0, 15e6)

        aim_value_index_i = aim_value_index_j = 0
        rel_error_strength = rel_error_young_modulus = 0.0

        for confining_pressure in self.confining_pressure_list:

            for texture_angle in self.texture_angle_list:

                X_test = [[confining_pressure, texture_angle, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]]
                X_test = run_ml_4.my_normalizer(X_test, low, up)

                predicted_strength = ML_xgb_4.predict(X_test)
                predicted_young_modulus = ML_xgb_5.predict(X_test)
                
                rel_error_strength += ((predicted_strength - self.aim_strength[aim_value_index_i][aim_value_index_j]) / self.aim_strength[aim_value_index_i][aim_value_index_j])**2
                if not (aim_value_index_i == 0 and aim_value_index_j == 1):
                    rel_error_young_modulus += ((predicted_young_modulus - self.aim_young_modulus[aim_value_index_i][aim_value_index_j]) / self.aim_young_modulus[aim_value_index_i][aim_value_index_j])**2
            
            aim_value_index_i += 1

        aim_value_index_j += 1

        if rel_error_strength + rel_error_young_modulus:
            fitness = 1 / (rel_error_strength + rel_error_young_modulus)
        else:
            fitness = 0.0

        return fitness
 
    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]
 
    def selection(self, individuals, k):
        """
        select some good individuals from pop, note that good individuals have greater probability to be choosen
        for example: a fitness list like that:[5, 4, 3, 2, 1], sum is 15,
        [-----|----|---|--|-]
        012345|6789|101112|1314|15
        we randomly choose a value in [0, 15],
        it belongs to first scale with greatest probability
        """
        s_inds = sorted(individuals, key=itemgetter("fitness"),
                        reverse=True)  # sort the pop by the reference of fitness
        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop
 
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']  # sum up the fitness
                if sum_ >= u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of
                    # [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # from small to large, due to list.pop() method get the last element
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen
 
    def crossoperate(self, offspring):
        """
        cross operation
        here we use two points crossoperate
        for example: gene1: [5, 2, 4, 7], gene2: [3, 6, 9, 2], if pos1=1, pos2=2
        5 | 2 | 4  7
        3 | 6 | 9  2
        =
        3 | 2 | 9  2
        5 | 6 | 4  7
        """
        dim = len(offspring[0]['Gene'].data)
 
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
 
        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
            pos2 = random.randrange(1, dim)
 
        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
 
        return newoff1, newoff2
 
    def mutation(self, crossoff, bound):
        """
        mutation operation
        """
        dim = len(crossoff.data)
 
        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation.
 
        if pos == 2 or pos == 5:
            temp_add = (bound[1][pos] - bound[0][pos]) * random.random() + bound[0][pos]
            crossoff.data[pos] = round(temp_add, 2)
        else:
            crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])
        return crossoff
    
    def GA_main(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3] + self.g * 2
 
        # Begin the evolution
        ############# ML part################
        data_min_list = self.parameter[4]
        data_min_list.insert(0, 0)
        data_min_list.insert(0, 0)
        data_max_list = self.parameter[5]
        data_max_list.insert(0, 90)
        data_max_list.insert(0, 15e6)

        #strength predictor
        predict_index = 14
        run_ml_4 = MachineLearning()
        ML_xgb_4 = run_ml_4.ML_main(data_min_list, data_max_list, predict_index, self.g)

        #Young's modulus predictor
        predict_index = 15
        run_ml_5 = MachineLearning()
        ML_xgb_5 = run_ml_5.ML_main(data_min_list, data_max_list, predict_index, self.g)

        # inside GA loop
        self.pop_in = self.pop
        self.bestindividual_in = self.bestindividual
        

        for g_in in range(NGEN):

            #self.log_export_file.write("############### Inside Generation {} ###############".format(g_in) + '\n')
            #self.log_export_file.flush()

            # Apply selection based on their converted fitness
            selectpop_in = self.selection(self.pop_in, popsize)

            nextoff_in = []
            while len(nextoff_in) != popsize:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                offspring_in = [selectpop_in.pop() for _ in range(2)]

                if random.random() < CXPB:  # cross two individuals with probability CXPB
                    crossoff1_in, crossoff2_in = self.crossoperate(offspring_in)
                    if random.random() < MUTPB:  # mutate an individual with probability MUTPB
                        muteoff1_in = self.mutation(crossoff1_in, self.bound)
                        muteoff2_in = self.mutation(crossoff2_in, self.bound)
                        fit_muteoff1_in = self.evaluate_in(muteoff1_in.data, ML_xgb_4, run_ml_4, ML_xgb_5)  # Evaluate the individuals
                        fit_muteoff2_in = self.evaluate_in(muteoff2_in.data, ML_xgb_4, run_ml_4, ML_xgb_5)  # Evaluate the individuals
                        nextoff_in.append({'Gene': muteoff1_in, 'fitness': fit_muteoff1_in})
                        nextoff_in.append({'Gene': muteoff2_in, 'fitness': fit_muteoff2_in})
                    else:
                        fit_crossoff1_in = self.evaluate_in(crossoff1_in.data, ML_xgb_4, run_ml_4, ML_xgb_5)  # Evaluate the individuals
                        fit_crossoff2_in = self.evaluate_in(crossoff2_in.data, ML_xgb_4, run_ml_4, ML_xgb_5)
                        nextoff_in.append({'Gene': crossoff1_in, 'fitness': fit_crossoff1_in})
                        nextoff_in.append({'Gene': crossoff2_in, 'fitness': fit_crossoff2_in})
                else:
                    nextoff_in.extend(offspring_in)

            # The population is entirely replaced by the offspring
            self.pop_in = nextoff_in

            self.best_ind_in = self.selectBest(self.pop_in)

            if self.best_ind_in['fitness'] > self.bestindividual_in[0]['fitness']:
                self.bestindividual_in = self.best_ind_in

        pop_file_name = "G" + str(self.g + 1) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        with open(pop_file_path, "a+") as f_w:
            indiv_ = self.bestindividual_in
            f_w.write(str(indiv_['Gene'].data[0])+ ' ' + str(indiv_['Gene'].data[1])+ ' ' + str(indiv_['Gene'].data[2])\
                    + ' ' + str(indiv_['Gene'].data[3]) + ' ' + str(indiv_['Gene'].data[4])+ ' ' + str(indiv_['Gene'].data[5]) \
                    + ' ' + str(indiv_['Gene'].data[6]) + ' ' + str(indiv_['Gene'].data[7])+ ' ' + str(indiv_['Gene'].data[8]) \
                    + ' ' + str(indiv_['Gene'].data[9]) + ' ' + str(indiv_['Gene'].data[10])+ ' ' + str(indiv_['Gene'].data[11]) + '\n')
            indiv_ = self.best_ind_in
            f_w.write(str(indiv_['Gene'].data[0])+ ' ' + str(indiv_['Gene'].data[1])+ ' ' + str(indiv_['Gene'].data[2])\
                    + ' ' + str(indiv_['Gene'].data[3]) + ' ' + str(indiv_['Gene'].data[4])+ ' ' + str(indiv_['Gene'].data[5]) \
                    + ' ' + str(indiv_['Gene'].data[6]) + ' ' + str(indiv_['Gene'].data[7])+ ' ' + str(indiv_['Gene'].data[8]) \
                    + ' ' + str(indiv_['Gene'].data[9]) + ' ' + str(indiv_['Gene'].data[10])+ ' ' + str(indiv_['Gene'].data[11]) + '\n')
        f_w.close()
            
if __name__ == "__main__":
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.2, 300, 200  # popsize must be even number
    #aim_strength, aim_young_modulus = 4.323e7, 5.54e9
    #aim_strain = 1.01265

    # Rows means 0,5,15 MPa
    # Columns means 0,45,90 degree
    # Value 0.0 means unreliable value
    aim_strength      = [[193.1e6, 0.75e6, 67.53e6],\
                         [208.47e6, 106.9e6, 121.77e6],\
                         [266.83e6, 58.9e6, 182.77e6]]
    
    aim_young_modulus = [[44.97e9, 0.0, 181.68e9],\
                         [29.14e9, 109.65e9, 99.94e9],\
                         [40.0e9, 62.65e9, 62.5e9]]
 
    #variables list[strong_p_E, strong_b_E, strong_b_knks, weak_p_E, weak_b_E, weak_b_knks,
    #               strong_b_n_max, strong_b_t_max, strong_b_phi, weak_b_n_max, weak_b_t_max, weak_b_phi]
    low = [5e8, 5e8, 1, 5e8, 5e8, 1, 2e6, 2e6, 0, 2e6, 2e6, 0]  # lower range for variables
    up  = [1e11, 1e11, 3, 1e11, 1e11, 3, 2e8, 2e8, 50, 2e8, 2e8, 50]  # upper range for variables
    parameter = [CXPB, MUTPB, NGEN, popsize, low, up, aim_strength, aim_young_modulus]
    run = GA(parameter)
    run.GA_main()