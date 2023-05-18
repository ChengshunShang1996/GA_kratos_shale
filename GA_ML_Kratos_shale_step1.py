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
 
        pop = []
        for i in range(self.parameter[3]):
            geneinfo = []
            for pos in range(len(low)):
                if pos == 2 or pos == 5: #to generate random float
                    temp_add = (self.bound[1][pos] - self.bound[0][pos]) * random.random() + self.bound[0][pos]
                    geneinfo.append(round(temp_add, 2))  # initialise popluation
                else:
                    geneinfo.append(random.randint(self.bound[0][pos], self.bound[1][pos]))  # initialise popluation
 
            fitness = self.evaluate(geneinfo)  # evaluate each chromosome 
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})  # store the chromosome and its fitness
            #pop.append({'Gene': Gene(data=geneinfo)})
        
        #for inin in pop:
        #    print(str(inin['Gene'].data[0]) + ' ' + str(inin['Gene'].data[1]) + ' ' + str(inin['Gene'].data[2]) + ' ' + str(inin['Gene'].data[3]))
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population
        self.clear_old_and_creat_new_kratos_data_folder() # clear old and creat new kratos data folder
        self.save_pop_and_bestindividual_into_file(0, self.pop, self.bestindividual)
    
    def evaluate(self, geneinfo):
        """
        fitness function
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
        y = 1 / ((x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2 + x10**2 + x11**2 + x12**2) * 1e5) #set the initial fitness a very small value
        return y
    
    def clear_old_and_creat_new_kratos_data_folder(self):

        kratos_data_folder_name = 'kratos_results_data'
        
        if os.path.exists(kratos_data_folder_name):
            shutil.rmtree(kratos_data_folder_name, ignore_errors=True)
            os.makedirs(kratos_data_folder_name)
        else:
            os.makedirs(kratos_data_folder_name)

    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]
    
    def save_pop_and_bestindividual_into_file(self, g, pop, best_individual):

        pop_file_name = "G" + str(g) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        bestindividual_file_name = "G" + str(g) + "_bestindividual.txt"
        bestindividual_file_path = os.path.join(os.getcwd(),'kratos_results_data', bestindividual_file_name)

        with open(pop_file_path, "a+") as f_w:
            for indiv_ in pop:
                f_w.write(str(indiv_['Gene'].data[0])+ ' ' + str(indiv_['Gene'].data[1])+ ' ' + str(indiv_['Gene'].data[2])\
                        + ' ' + str(indiv_['Gene'].data[3]) + ' ' + str(indiv_['Gene'].data[4])+ ' ' + str(indiv_['Gene'].data[5]) \
                        + ' ' + str(indiv_['Gene'].data[6]) + ' ' + str(indiv_['Gene'].data[7])+ ' ' + str(indiv_['Gene'].data[8]) \
                        + ' ' + str(indiv_['Gene'].data[9]) + ' ' + str(indiv_['Gene'].data[10])+ ' ' + str(indiv_['Gene'].data[11]) + '\n')
        f_w.close()

        with open(bestindividual_file_path, "a+") as f_w:
            f_w.write(str(best_individual['Gene'].data[0])+ ' ' + str(best_individual['Gene'].data[1])+ ' ' + str(best_individual['Gene'].data[2])\
                      + ' ' + str(best_individual['Gene'].data[3]) + ' ' + str(best_individual['Gene'].data[4])+ ' ' + str(best_individual['Gene'].data[5]) \
                      + ' ' + str(best_individual['Gene'].data[6]) + ' ' + str(best_individual['Gene'].data[7])+ ' ' + str(best_individual['Gene'].data[8]) \
                      + ' ' + str(best_individual['Gene'].data[9]) + ' ' + str(best_individual['Gene'].data[10])+ ' ' + str(best_individual['Gene'].data[11]) \
                      + ' ' + str(best_individual['fitness'])+ '\n')
        f_w.close()

    def GA_main(self):
        pass

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