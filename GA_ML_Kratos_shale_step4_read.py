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

        self.g = 0
        g = self.g
 
        pop = []
        bestindividual_read = []

        pop_file_name = "G" + str(g) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        with open(pop_file_path, "r") as f_r:
            for line in f_r:
                geneinfo = []
                values = [float(s) for s in line.split()]
                for cnt in range(0,12):
                    geneinfo.append(values[cnt])
                pop.append({'Gene': Gene(data=geneinfo), 'fitness': 0.0})     
        f_r.close()

        if g == 0:
            bestindividual_file_name = "G" + str(g) + "_bestindividual.txt"
        else:
            bestindividual_file_name = "G" + str(g-1) + "_bestindividual.txt"
        bestindividual_file_path = os.path.join(os.getcwd(),'kratos_results_data', bestindividual_file_name)

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

    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]
    
    def read_kratos_results_and_add_fitness(self, g_count, nextoff):
        
        for indiv_ in nextoff:

            strong_p_E      = str(indiv_['Gene'].data[0])
            strong_b_E      = str(indiv_['Gene'].data[1])
            strong_b_knks   = str(indiv_['Gene'].data[2])
            weak_p_E        = str(indiv_['Gene'].data[3])
            weak_b_E        = str(indiv_['Gene'].data[4])
            weak_b_knks     = str(indiv_['Gene'].data[5])
            strong_b_n_max  = str(indiv_['Gene'].data[6])
            strong_b_t_max  = str(indiv_['Gene'].data[7])
            strong_b_phi    = str(indiv_['Gene'].data[8])
            weak_b_n_max    = str(indiv_['Gene'].data[9])
            weak_b_t_max    = str(indiv_['Gene'].data[10])
            weak_b_phi      = str(indiv_['Gene'].data[11])

            rel_error_strength = rel_error_young_modulus = 0.0
            aim_value_index_i = aim_value_index_j = 0

            #write out individual data for ML
            output_file_name = 'G' + str(g_count) + '_info.csv' 
            output_aim_path_and_name = os.path.join(os.getcwd(),'kratos_results_data', output_file_name)

            indiv_data_head = ['confining_pressure','texture_angle','strong_p_E', 'strong_b_E', 'strong_b_knks', \
                               'weak_p_E', 'weak_b_E', 'weak_b_knks', 'strong_b_n_max', 'strong_b_t_max', \
                                'strong_b_phi', 'weak_b_n_max', 'weak_b_t_max', 'weak_b_phi', 'strength_max', 'young_modulus_max']

            for confining_pressure in self.confining_pressure_list:

                for texture_angle in self.texture_angle_list:

                    #creat new folder
                    aim_folder_name = 'G' + str(g_count) + '_' + str(confining_pressure) + '_' + str(texture_angle) + '_' \
                                        + strong_p_E + '_' + strong_b_E + '_' + strong_b_knks + '_'\
                                        + weak_p_E + '_' + weak_b_E + '_' + weak_b_knks + '_'\
                                        + strong_b_n_max + '_' + strong_b_t_max + '_' + strong_b_phi + '_'\
                                        + weak_b_n_max + '_' + weak_b_t_max + '_' + weak_b_phi
                    #the strength files
                    aim_path_and_name = os.path.join(os.getcwd(),'Generated_kratos_cases', aim_folder_name, 'G-Triaxial_Graphs', 'G-Triaxial_graph.grf')

                    if os.path.getsize(aim_path_and_name) != 0:
                        stress_data_list = []
                        #strain_data_list = []
                        with open(aim_path_and_name, 'r') as stress_strain_data:
                            for line in stress_strain_data:
                                values = [float(s) for s in line.split()]
                                stress_data_list.append(values[1]) 
                                #strain_data_list.append(values[0])
                        strength_max = max(stress_data_list)
                        #strain_max = strain_data_list[stress_data_list.index(max(stress_data_list))]
                        rel_error_strength += ((strength_max - self.aim_strength[aim_value_index_i][aim_value_index_j]) / self.aim_strength[aim_value_index_i][aim_value_index_j])**2
                        #rel_error_starin   += ((strain_max - self.aim_strain) / self.aim_strain)**2
                    else:
                        strength_max = 0.0
                        #strain_max = 0.0
                        rel_error_strength += (self.aim_strength[aim_value_index_i][aim_value_index_j])**2
                        #rel_error_starin   += (self.aim_strain)**2

                    #the Young modulus files
                    aim_path_and_name = os.path.join(os.getcwd(),'Generated_kratos_cases', aim_folder_name, 'G-Triaxial_Graphs', 'G-Triaxial_graph_young.grf')

                    if os.path.getsize(aim_path_and_name) != 0:
                        strain_data_list = []
                        young_data_list = []
                        with open(aim_path_and_name, 'r') as young_data:
                            for line in young_data:
                                values = [float(s) for s in line.split()]
                                strain_data_list.append(values[0])
                                young_data_list.append(values[1]) 
                        strain_max = strain_data_list[stress_data_list.index(max(stress_data_list))]
                        if max(strain_data_list) > 0.2:
                            young_cnt = 0
                            young_select_sum = 0.0
                            for strain_data in strain_data_list:
                                if strain_data > 0.3 * strain_max and strain_data < 0.5 * strain_max:
                                    young_select_sum += young_data_list[strain_data_list.index(strain_data)]
                                    young_cnt += 1
                            young_modulus_max = young_select_sum / young_cnt
                        else:
                            young_modulus_max = max(young_data_list)

                        if not (aim_value_index_i == 0 and aim_value_index_j == 1):
                            rel_error_young_modulus += ((young_modulus_max - self.aim_young_modulus[aim_value_index_i][aim_value_index_j]) / self.aim_young_modulus[aim_value_index_i][aim_value_index_j])**2
                    else:
                        young_modulus_max = 0.0

                        if not (aim_value_index_i == 0 and aim_value_index_j == 1):
                            rel_error_young_modulus += (self.aim_young_modulus[aim_value_index_i][aim_value_index_j])**2

                    #write out indiv information
                    indiv_data = []
                    with open(output_aim_path_and_name, "a+", encoding='UTF8', newline='') as f_w:
                        writer = csv.writer(f_w)
                        indiv_data.append(confining_pressure)
                        indiv_data.append(texture_angle)
                        indiv_data.append(strong_p_E)
                        indiv_data.append(strong_b_E)
                        indiv_data.append(strong_b_knks)
                        indiv_data.append(weak_p_E)
                        indiv_data.append(weak_b_E)
                        indiv_data.append(weak_b_knks)
                        indiv_data.append(strong_b_n_max)
                        indiv_data.append(strong_b_t_max)
                        indiv_data.append(strong_b_phi)
                        indiv_data.append(weak_b_n_max)
                        indiv_data.append(weak_b_t_max)
                        indiv_data.append(weak_b_phi)
                        indiv_data.append(strength_max)
                        #indiv_data.append(strain_max)
                        indiv_data.append(young_modulus_max)
                        if self.indiv_data_head_not_written:
                            writer.writerow(indiv_data_head)
                            self.indiv_data_head_not_written = False
                        writer.writerow(indiv_data)
                        f_w.close()

                aim_value_index_j += 1
            aim_value_index_i += 1

            if rel_error_strength + rel_error_young_modulus:
                fitness = 1 / (rel_error_strength + rel_error_young_modulus)
            else:
                fitness = 0

            indiv_['fitness'] = fitness

        return nextoff
    
    def save_and_plot_best_individual_results(self, g_count, best_individual):
        
        #save data files
        bestindividual_file_name = "G" + str(g_count) + "_bestindividual.txt"
        bestindividual_file_path = os.path.join(os.getcwd(),'kratos_results_data', bestindividual_file_name)

        with open(bestindividual_file_path, "a+") as f_w:
            f_w.write(str(best_individual['Gene'].data[0])+ ' ' + str(best_individual['Gene'].data[1])+ ' ' + str(best_individual['Gene'].data[2])\
                      + ' ' + str(best_individual['Gene'].data[3]) + ' ' + str(best_individual['Gene'].data[4])+ ' ' + str(best_individual['Gene'].data[5]) \
                      + ' ' + str(best_individual['Gene'].data[6]) + ' ' + str(best_individual['Gene'].data[7])+ ' ' + str(best_individual['Gene'].data[8]) \
                      + ' ' + str(best_individual['Gene'].data[9]) + ' ' + str(best_individual['Gene'].data[10])+ ' ' + str(best_individual['Gene'].data[11]) \
                      + ' ' + str(best_individual['fitness'])+ '\n')
        f_w.close()
        
        #plot and save
        strong_p_E      = str(best_individual['Gene'].data[0])
        strong_b_E      = str(best_individual['Gene'].data[1])
        strong_b_knks   = str(best_individual['Gene'].data[2])
        weak_p_E        = str(best_individual['Gene'].data[3])
        weak_b_E        = str(best_individual['Gene'].data[4])
        weak_b_knks     = str(best_individual['Gene'].data[5])
        strong_b_n_max  = str(best_individual['Gene'].data[6])
        strong_b_t_max  = str(best_individual['Gene'].data[7])
        strong_b_phi    = str(best_individual['Gene'].data[8])
        weak_b_n_max    = str(best_individual['Gene'].data[9])
        weak_b_t_max    = str(best_individual['Gene'].data[10])
        weak_b_phi      = str(best_individual['Gene'].data[11])

        for confining_pressure in self.confining_pressure_list:

            for texture_angle in self.texture_angle_list:

                #creat new folder
                from_folder_name = 'G' + str(g_count) + '_' + str(confining_pressure) + '_' + str(texture_angle) + '_' \
                                    + strong_p_E + '_' + strong_b_E + '_' + strong_b_knks + '_'\
                                    + weak_p_E + '_' + weak_b_E + '_' + weak_b_knks + '_'\
                                    + strong_b_n_max + '_' + strong_b_t_max + '_' + strong_b_phi + '_'\
                                    + weak_b_n_max + '_' + weak_b_t_max + '_' + weak_b_phi

                to_folder_name = 'G_' + str(g_count) + '_' + str(confining_pressure) + '_' + str(texture_angle)
                #save the best individual case to results folder
                from_directory = os.path.join(os.getcwd(),'Generated_kratos_cases', from_folder_name)
                to_directory = os.path.join(os.getcwd(),'kratos_results_data', to_folder_name)
                copy_tree(from_directory, to_directory)
    
    def GA_main(self):
        """
        main frame work of GA
        """

        #add fitness to nextoff
        nextoff = self.read_kratos_results_and_add_fitness(self.g, self.pop)

        # The population is entirely replaced by the offspring
        self.pop = nextoff

        best_ind = self.selectBest(self.pop)

        if best_ind['fitness'] > self.bestindividual[0]['fitness']:
            self.bestindividual = best_ind
            # save the data of the best individual for post processing
            self.save_and_plot_best_individual_results(self.g, self.bestindividual)
        else:
            self.save_and_plot_best_individual_results(self.g, best_ind)
 
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