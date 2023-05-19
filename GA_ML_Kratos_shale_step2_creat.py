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

        if g == 1:
            pop_file_name = "G" + str(g-1) + "_pop.txt"
        else:
            pop_file_name = "G" + str(g) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        if g == 0:
            bestindividual_file_name = "G" + str(g) + "_bestindividual.txt"
        else:
            bestindividual_file_name = "G" + str(g-1) + "_bestindividual.txt"
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
        
        #for inin in pop:
        #    print(str(inin['Gene'].data[0]) + ' ' + str(inin['Gene'].data[1]) + ' ' + str(inin['Gene'].data[2]) + ' ' + str(inin['Gene'].data[3]))
        self.pop = pop
        self.bestindividual = bestindividual_read  # store the best chromosome in the population
        #self.clear_old_and_creat_new_kratos_data_folder() # clear old and creat new kratos data folder
        self.aim_strength = parameter[6]
        self.aim_young_modulus = parameter[7]
        self.indiv_data_head_not_written = True
        self.confining_pressure_list = [0, 5e6, 15e6]
        self.texture_angle_list = [0, 45, 90]
 
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
    
    def clear_old_and_creat_new_kratos_data_folder(self):

        kratos_data_folder_name = 'kratos_results_data'
        
        if os.path.exists(kratos_data_folder_name):
            shutil.rmtree(kratos_data_folder_name, ignore_errors=True)
            os.makedirs(kratos_data_folder_name)
        else:
            os.makedirs(kratos_data_folder_name)
    
    def clear_old_and_creat_new_kratos_case_folder(self):

        kratos_case_folder_name = 'Generated_kratos_cases'

        if os.path.exists(kratos_case_folder_name):
            shutil.rmtree(kratos_case_folder_name, ignore_errors=True)
            os.makedirs(kratos_case_folder_name)
        else:
            os.makedirs(kratos_case_folder_name)

        kratos_temp_data_folder_name = 'kratos_results_data_temp'
        
        if os.path.exists(kratos_temp_data_folder_name):
            shutil.rmtree(kratos_temp_data_folder_name, ignore_errors=True)
            os.makedirs(kratos_temp_data_folder_name)
        else:
            os.makedirs(kratos_temp_data_folder_name)

        cases_run_path_and_name = os.path.join(os.getcwd(),'cases_run.sh')
        if os.path.exists(cases_run_path_and_name):
            os.remove(cases_run_path_and_name)
    
    def clear_old_out_and_err_files(self):

        for item in os.listdir('.'):
            if item.endswith(".err"):
                os.remove(item)
            elif item.endswith(".out"):
                os.remove(item)

    def clear_old_out_files(self):

        my_name = 'm'

        for f in os.listdir('.'):
            if any(x in f for x in my_name) and f.endswith('.out'):
                os.remove(f)

    def save_next_pop_into_file(self, g, pop):

        pop_file_name = "G" + str(g+1) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        with open(pop_file_path, "a+") as f_w:
            for indiv_ in pop:
                f_w.write(str(indiv_['Gene'].data[0])+ ' ' + str(indiv_['Gene'].data[1])+ ' ' + str(indiv_['Gene'].data[2])\
                        + ' ' + str(indiv_['Gene'].data[3]) + ' ' + str(indiv_['Gene'].data[4])+ ' ' + str(indiv_['Gene'].data[5]) \
                        + ' ' + str(indiv_['Gene'].data[6]) + ' ' + str(indiv_['Gene'].data[7])+ ' ' + str(indiv_['Gene'].data[8]) \
                        + ' ' + str(indiv_['Gene'].data[9]) + ' ' + str(indiv_['Gene'].data[10])+ ' ' + str(indiv_['Gene'].data[11]) +'\n')
        f_w.close()

    def add_pop_end_to_current_pop(self, g):
        
        if g == 1: 
            pop_file_name = "G" + str(g-1) + "_pop.txt"
        else:
            pop_file_name = "G" + str(g) + "_pop.txt"
        pop_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_file_name)

        pop_end_file_name = "G" + str(g) + "_pop_end.txt"
        pop_end_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_end_file_name)

        with open(pop_end_file_path) as f_r:
            with open(pop_file_path, "a+") as f_w:
                for line in f_r:
                    f_w.write(line)

    def uniquify(self, path, path_change_marker):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + "_" + str(counter) + extension
            counter += 1
            path_change_marker += 1

        return path, path_change_marker
    
    def generate_kratos_cases(self, g_count, nextoff):

        #self.log_export_file.write('Generating kratos cases ...' + '\n')
        #self.log_export_file.flush()
        self.end_sim_file_num = 0
        self.is_sh_head_write = False
        self.sh_marker = 0

        #loop every individual in the pop
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

            for confining_pressure in self.confining_pressure_list:

                for texture_angle in self.texture_angle_list:

                    if confining_pressure == 0: #confining pressure = 0.0

                        #creat new folder
                        new_folder_name = 'G' + str(g_count) + '_' + str(confining_pressure) + '_' + str(texture_angle) + '_' \
                                            + strong_p_E + '_' + strong_b_E + '_' + strong_b_knks + '_'\
                                            + weak_p_E + '_' + weak_b_E + '_' + weak_b_knks + '_'\
                                            + strong_b_n_max + '_' + strong_b_t_max + '_' + strong_b_phi + '_'\
                                            + weak_b_n_max + '_' + weak_b_t_max + '_' + weak_b_phi
                        aim_path = os.path.join(os.getcwd(),'Generated_kratos_cases', new_folder_name)
                        aim_path_change_marker = 0
                        aim_path, aim_path_change_marker = self.uniquify(aim_path, aim_path_change_marker)
                        os.mkdir(aim_path)

                        #copy source file
                        texture_angle_folder = 'angle_' + str(texture_angle)
                        seed_file_name_list = ['decompressed_material_triaxial_test_PBM_GA_230413.py', 'G-TriaxialDEM_FEM_boundary.mdpa',\
                                                'G-TriaxialDEM.mdpa', 'ProjectParametersDEM.json', 'MaterialsDEM.json', 'run_omp.sh']
                        for seed_file_name in seed_file_name_list:
                            seed_file_path_and_name = os.path.join(os.getcwd(), 'kratos_seed_files', 'UCS', texture_angle_folder, seed_file_name)
                            aim_file_path_and_name = os.path.join(aim_path, seed_file_name)

                            if seed_file_name == 'MaterialsDEM.json':
                                young_modulus_marker = 0
                                bond_young_modulus_marker = 0
                                bond_knks_marker = 0
                                bond_sigma_max_marker = 0
                                bond_tau_zero_marker = 0
                                bond_phi_marker = 0
                                with open(seed_file_path_and_name, "r") as f_material:
                                    with open(aim_file_path_and_name, "w") as f_material_w:
                                        for line in f_material.readlines():
                                            if "YOUNG_MODULUS" in line:
                                                if young_modulus_marker == 0:
                                                    line = line.replace("74.95e9", str(strong_p_E))
                                                    young_modulus_marker += 1
                                                elif young_modulus_marker == 1:
                                                    line = line.replace("74.95e9", str(strong_p_E))
                                                    young_modulus_marker += 1
                                                elif young_modulus_marker == 2:
                                                    line = line.replace("36.93e9", str(weak_p_E))
                                                    young_modulus_marker += 1
                                            elif "BOND_YOUNG_MODULUS" in line:
                                                if bond_young_modulus_marker == 0:
                                                    line = line.replace("74.95e9", str(strong_b_E))
                                                    bond_young_modulus_marker += 1
                                                elif bond_young_modulus_marker == 1:
                                                    line = line.replace("74.95e9", str(strong_b_E))
                                                    bond_young_modulus_marker += 1
                                                elif bond_young_modulus_marker == 2:
                                                    line = line.replace("36.93e9", str(weak_b_E))
                                                    bond_young_modulus_marker += 1
                                            elif "BOND_KNKS_RATIO" in line:
                                                if bond_knks_marker == 0:
                                                    line = line.replace("2.5", str(strong_b_knks))
                                                    bond_knks_marker += 1
                                                elif bond_knks_marker == 1:
                                                    line = line.replace("2.5", str(strong_b_knks))
                                                    bond_knks_marker += 1
                                                elif bond_knks_marker == 2:
                                                    line = line.replace("2.5", str(weak_b_knks))
                                                    bond_knks_marker += 1
                                            elif "BOND_SIGMA_MAX" in line:
                                                if bond_sigma_max_marker == 0:
                                                    line = line.replace("198.43e7", str(strong_b_n_max))
                                                elif bond_sigma_max_marker == 1:
                                                    line = line.replace("198.43e7", str(strong_b_n_max))
                                                elif bond_sigma_max_marker == 2:
                                                    line = line.replace("54.7e6", str(weak_b_n_max))
                                            elif "BOND_TAU_ZERO" in line:
                                                if bond_tau_zero_marker == 0:
                                                    line = line.replace("40e7", str(strong_b_t_max))
                                                elif bond_tau_zero_marker == 1:
                                                    line = line.replace("40e6", str(strong_b_t_max))
                                                elif bond_tau_zero_marker == 2:
                                                    line = line.replace("11e6", str(weak_b_t_max))
                                            elif "BOND_INTERNAL_FRICC" in line:
                                                if bond_phi_marker == 0:
                                                    line = line.replace("0.0", str(strong_b_phi))
                                                elif bond_phi_marker == 1:
                                                    line = line.replace("0.0", str(strong_b_phi))
                                                elif bond_phi_marker == 2:
                                                    line = line.replace("0.0", str(weak_b_phi))
                                            f_material_w.write(line)
                            elif seed_file_name == 'ProjectParametersDEM.json':
                                with open(seed_file_path_and_name, "r") as f_parameter:
                                    with open(aim_file_path_and_name, "w") as f_parameter_w:
                                        for line in f_parameter.readlines():
                                            f_parameter_w.write(line)
                            elif seed_file_name == 'run_omp.sh':
                                with open(seed_file_path_and_name, "r") as f_run_omp:
                                    with open(aim_file_path_and_name, "w") as f_run_omp_w:
                                        for line in f_run_omp.readlines():
                                            if "BTS-Q-Ep6.2e10-T1e3-f0.1" in line:
                                                hpc_case_name = os.path.basename(aim_path)
                                                line = line.replace("BTS-Q-Ep6.2e10-T1e3-f0.1", hpc_case_name)
                                            f_run_omp_w.write(line)
                            else:
                                shutil.copyfile(seed_file_path_and_name, aim_file_path_and_name) 

                        # write the cases_run.sh
                        if aim_path_change_marker == 0:
                            nodes_num = 16
                            #partition_name_list = ['B510','HM','HighParallelization','R182-open']
                            partition_name_list = ['B510','HM','R182-open']
                            partition_name = random.choice(partition_name_list)
                            self.end_sim_file_num += 1
                            new_sh_marker = (self.end_sim_file_num - 1) // nodes_num
                            if new_sh_marker > self.sh_marker:
                                self.sh_marker = new_sh_marker
                                self.is_sh_head_write = False
                            sh_file_name = 'cases_run_' + str(self.sh_marker) + '.sh'

                            # creat the cases_run.sh
                            cases_run_path_and_name = os.path.join(os.getcwd(), 'kratos_results_data_temp', sh_file_name)

                            with open(cases_run_path_and_name, "a") as f_w_cases_run:
                                if self.is_sh_head_write == False:
                                    f_w_cases_run.write('#!/bin/bash'+'\n')
                                    f_w_cases_run.write('#SBATCH --job-name=Generation_'+ str(g_count) + '_part_'+ str(self.sh_marker) +'\n')
                                    f_w_cases_run.write('#SBATCH --output=m_chengshun_job%j.out'+'\n')
                                    f_w_cases_run.write('#SBATCH --error=m_chengshun_job%j.err'+'\n')
                                    f_w_cases_run.write('#SBATCH --partition='+ partition_name +'\n')
                                    f_w_cases_run.write('#SBATCH --ntasks-per-node='+str(nodes_num)+'\n')
                                    #f_w_cases_run.write('#SBATCH --nodes=1'+'\n'+'\n')
                                    self.is_sh_head_write = True
                                f_w_cases_run.write('cd '+ aim_path + '\n')
                                f_w_cases_run.write('python3 '+ 'decompressed_material_triaxial_test_PBM_GA_230413.py' + '\n')
                            f_w_cases_run.close()

                    else: #confining pressure > 0.0

                        #creat new folder
                        new_folder_name = 'G' + str(g_count) + '_' + str(confining_pressure) + '_' + str(texture_angle) + '_' \
                                            + strong_p_E + '_' + strong_b_E + '_' + strong_b_knks + '_'\
                                            + weak_p_E + '_' + weak_b_E + '_' + weak_b_knks + '_'\
                                            + strong_b_n_max + '_' + strong_b_t_max + '_' + strong_b_phi + '_'\
                                            + weak_b_n_max + '_' + weak_b_t_max + '_' + weak_b_phi
                        aim_path = os.path.join(os.getcwd(),'Generated_kratos_cases', new_folder_name)
                        aim_path_change_marker = 0
                        aim_path, aim_path_change_marker = self.uniquify(aim_path, aim_path_change_marker)
                        os.mkdir(aim_path)

                        #copy source file
                        texture_angle_folder = 'angle_' + str(texture_angle)
                        seed_file_name_list = ['dem_wrapper_cshang_230424.py', 'FEM_membrane.mdpa','G-TriaxialDEM.mdpa',\
                                                'G-TriaxialDEM_FEM_boundary.mdpa', 'MainKratos_230424.py', 'MaterialsDEM.json', \
                                                'ProjectParametersCoSim.json', 'ProjectParametersDEM.json', 'ProjectParametersFEM.json',\
                                                'StructuralMaterials.json','gauss_seidel_weak_cshang_230424.py']
                        for seed_file_name in seed_file_name_list:
                            seed_file_path_and_name = os.path.join(os.getcwd(),'kratos_seed_files','Triaxial', texture_angle_folder, seed_file_name)
                            aim_file_path_and_name = os.path.join(aim_path, seed_file_name)

                            if seed_file_name == 'MaterialsDEM.json':
                                young_modulus_marker = 0
                                bond_young_modulus_marker = 0
                                bond_knks_marker = 0
                                bond_sigma_max_marker = 0
                                bond_tau_zero_marker = 0
                                bond_phi_marker = 0
                                with open(seed_file_path_and_name, "r") as f_material:
                                    with open(aim_file_path_and_name, "w") as f_material_w:
                                        for line in f_material.readlines():
                                            if "YOUNG_MODULUS" in line:
                                                if young_modulus_marker == 0:
                                                    line = line.replace("74.95e9", str(strong_p_E))
                                                    young_modulus_marker += 1
                                                elif young_modulus_marker == 1:
                                                    line = line.replace("74.95e9", str(strong_p_E))
                                                    young_modulus_marker += 1
                                                elif young_modulus_marker == 2:
                                                    line = line.replace("36.93e9", str(weak_p_E))
                                                    young_modulus_marker += 1
                                            elif "BOND_YOUNG_MODULUS" in line:
                                                if bond_young_modulus_marker == 0:
                                                    line = line.replace("74.95e9", str(strong_b_E))
                                                    bond_young_modulus_marker += 1
                                                elif bond_young_modulus_marker == 1:
                                                    line = line.replace("74.95e9", str(strong_b_E))
                                                    bond_young_modulus_marker += 1
                                                elif bond_young_modulus_marker == 2:
                                                    line = line.replace("36.93e9", str(weak_b_E))
                                                    bond_young_modulus_marker += 1
                                            elif "BOND_KNKS_RATIO" in line:
                                                if bond_knks_marker == 0:
                                                    line = line.replace("2.5", str(strong_b_knks))
                                                    bond_knks_marker += 1
                                                elif bond_knks_marker == 1:
                                                    line = line.replace("2.5", str(strong_b_knks))
                                                    bond_knks_marker += 1
                                                elif bond_knks_marker == 2:
                                                    line = line.replace("2.5", str(weak_b_knks))
                                                    bond_knks_marker += 1
                                            elif "BOND_SIGMA_MAX" in line:
                                                if bond_sigma_max_marker == 0:
                                                    line = line.replace("198.43e7", str(strong_b_n_max))
                                                elif bond_sigma_max_marker == 1:
                                                    line = line.replace("198.43e7", str(strong_b_n_max))
                                                elif bond_sigma_max_marker == 2:
                                                    line = line.replace("54.7e6", str(weak_b_n_max))
                                            elif "BOND_TAU_ZERO" in line:
                                                if bond_tau_zero_marker == 0:
                                                    line = line.replace("40e7", str(strong_b_t_max))
                                                elif bond_tau_zero_marker == 1:
                                                    line = line.replace("40e6", str(strong_b_t_max))
                                                elif bond_tau_zero_marker == 2:
                                                    line = line.replace("11e6", str(weak_b_t_max))
                                            elif "BOND_INTERNAL_FRICC" in line:
                                                if bond_phi_marker == 0:
                                                    line = line.replace("0.0", str(strong_b_phi))
                                                elif bond_phi_marker == 1:
                                                    line = line.replace("0.0", str(strong_b_phi))
                                                elif bond_phi_marker == 2:
                                                    line = line.replace("0.0", str(weak_b_phi))
                                            f_material_w.write(line)
                            elif seed_file_name == 'ProjectParametersFEM.json':
                                with open(seed_file_path_and_name, "r") as f_parameter:
                                    with open(aim_file_path_and_name, "w") as f_parameter_w:
                                        for line in f_parameter.readlines():
                                            if "-1000000.0*t" in line:
                                                line = line.replace("-1000000.0", str(confining_pressure))
                                            f_parameter_w.write(line)
                            else:
                                shutil.copyfile(seed_file_path_and_name, aim_file_path_and_name)

                        # write the cases_run.sh
                        if aim_path_change_marker == 0:
                            nodes_num = 16
                            #partition_name_list = ['B510','HM','HighParallelization','R182-open']
                            partition_name_list = ['B510','HM','R182-open']
                            partition_name = random.choice(partition_name_list)
                            self.end_sim_file_num += 1
                            new_sh_marker = (self.end_sim_file_num - 1) // nodes_num
                            if new_sh_marker > self.sh_marker:
                                self.sh_marker = new_sh_marker
                                self.is_sh_head_write = False
                            sh_file_name = 'cases_run_' + str(self.sh_marker) + '.sh'

                            # creat the cases_run.sh
                            cases_run_path_and_name = os.path.join(os.getcwd(), 'kratos_results_data_temp', sh_file_name)

                            with open(cases_run_path_and_name, "a") as f_w_cases_run:
                                if self.is_sh_head_write == False:
                                    f_w_cases_run.write('#!/bin/bash'+'\n')
                                    f_w_cases_run.write('#SBATCH --job-name=Generation_'+ str(g_count) + '_part_'+ str(self.sh_marker) +'\n')
                                    f_w_cases_run.write('#SBATCH --output=m_chengshun_job%j.out'+'\n')
                                    f_w_cases_run.write('#SBATCH --error=m_chengshun_job%j.err'+'\n')
                                    f_w_cases_run.write('#SBATCH --partition='+ partition_name +'\n')
                                    f_w_cases_run.write('#SBATCH --ntasks-per-node='+str(nodes_num)+'\n')
                                    self.is_sh_head_write = True
                                f_w_cases_run.write('cd '+ aim_path + '\n')
                                f_w_cases_run.write('python3 '+ 'MainKratos_230424.py' + '\n')
                            f_w_cases_run.close()
    
    
    def GA_main(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3] + 2 * self.g

        start_time = time.time()
 
        # Begin the evolution
 
        #clear old kratos case files and creat new one
        self.clear_old_and_creat_new_kratos_case_folder()
        #self.clear_old_out_and_err_files()
        self.clear_old_out_files()

        g = self.g
        if g != 0:
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                offspring = [selectpop.pop() for _ in range(2)]

                if random.random() < CXPB:  # cross two individuals with probability CXPB
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < MUTPB:  # mutate an individual with probability MUTPB
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        #fit_muteoff1 = self.evaluate(muteoff1.data)  # Evaluate the individuals
                        #fit_muteoff2 = self.evaluate(muteoff2.data)  # Evaluate the individuals
                        #nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        #nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                        nextoff.append({'Gene': muteoff1})
                        nextoff.append({'Gene': muteoff2})
                    else:
                        #fit_crossoff1 = self.evaluate(crossoff1.data)  # Evaluate the individuals
                        #fit_crossoff2 = self.evaluate(crossoff2.data)
                        #nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        #nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                        nextoff.append({'Gene': crossoff1})
                        nextoff.append({'Gene': crossoff2})
                else:
                    nextoff.extend(offspring)
            self.save_next_pop_into_file(g, nextoff)
            self.add_pop_end_to_current_pop(g)
            #the predicted best individual by inside GA are added to the population
            newoff1_add = Gene(data=[])  # offspring1 produced by cross operation
            newoff2_add = Gene(data=[])  # offspring2 produced by cross operation

            pop_end = []
            pop_end_file_name = "G" + str(g) + "_pop_end.txt"
            pop_end_file_path = os.path.join(os.getcwd(),'kratos_results_data', pop_end_file_name)

            with open(pop_end_file_path, "r") as f_r:
                for line in f_r:
                    geneinfo = []
                    values = [float(s) for s in line.split()]
                    for cnt in range(0,12):
                        geneinfo.append(values[cnt])
                    pop_end.append({'Gene': Gene(data=geneinfo), 'fitness': 0.0})     
            f_r.close()

            newoff1_add.data = pop_end[0]['Gene'].data
            newoff2_add.data = pop_end[1]['Gene'].data
            nextoff.append({'Gene': newoff1_add})
            nextoff.append({'Gene': newoff2_add})         
        else:
            nextoff = self.pop

        #generate kratos cases according to pop 
        self.generate_kratos_cases(g, nextoff)

 
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