import random
from operator import itemgetter
import os
import shutil
import glob
from distutils.dir_util import copy_tree
import time
import csv
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
                geneinfo.append(random.randint(self.bound[0][pos], self.bound[1][pos]))  # initialise popluation
 
            fitness = self.evaluate(geneinfo)  # evaluate each chromosome 
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})  # store the chromosome and its fitness
            #pop.append({'Gene': Gene(data=geneinfo)})
        
        #for inin in pop:
        #    print(str(inin['Gene'].data[0]) + ' ' + str(inin['Gene'].data[1]) + ' ' + str(inin['Gene'].data[2]) + ' ' + str(inin['Gene'].data[3]))
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population
        self.clear_old_and_creat_new_kratos_data_folder() # clear old and creat new kratos data folder
        self.aim_strength = parameter[6]
        self.aim_young_modulus = parameter[7]
        self.aim_strain = parameter[8]
        self.indiv_data_head_not_written = True
        self.confining_pressure_list = [0, 5, 15]
        self.texture_angle_list = [0, 45, 90]
 
    def evaluate(self, geneinfo):
        """
        fitness function
        """
        x1 = geneinfo[0]
        x2 = geneinfo[1]
        x3 = geneinfo[2]
        x4 = geneinfo[3]
        y = 1 / ((x1**2 + x2**2 + x3**2 + x4**2) * 1e5) #set the initial fitness a very small value
        return y
    
    def evaluate_in(self, geneinfo, ML_xgb_4, run_ml_4, ML_xgb_5, ML_xgb_6):
        """
        fitness function using ML model
        """
        x1 = geneinfo[0]
        x2 = geneinfo[1]
        x3 = geneinfo[2]
        x4 = geneinfo[3]
        X_test = [[x1,x2,x3,x4]]
        X_test = run_ml_4.my_normalizer(X_test, self.parameter[4], self.parameter[5])

        predicted_strength = ML_xgb_4.predict(X_test)
        predicted_starin = ML_xgb_5.predict(X_test)
        predicted_young_modulus = ML_xgb_6.predict(X_test)

        rel_error_strength = ((predicted_strength - self.aim_strength) / self.aim_strength)**2
        rel_error_starin   = ((predicted_starin - self.aim_strain) / self.aim_strain)**2
        rel_error_young_modulus = ((predicted_young_modulus - self.aim_young_modulus) / self.aim_young_modulus)**2

        if rel_error_strength + rel_error_young_modulus + rel_error_starin:
            fitness = 1 / (rel_error_strength + rel_error_young_modulus + rel_error_starin)
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

    def clear_old_and_creat_new_log_file(self):
        
        log_file_name = 'running_log.txt'
        log_output_path_and_name = os.path.join(os.getcwd(),'kratos_results_data', log_file_name)
        if os.path.exists(log_output_path_and_name):
            os.remove(log_output_path_and_name)
        log_export_file = open(log_output_path_and_name, 'a+')
        return log_export_file

    def uniquify(self, path, path_change_marker):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + "_" + str(counter) + extension
            counter += 1
            path_change_marker += 1

        return path, path_change_marker
    
    def generate_kratos_cases(self, g_count, nextoff):

        self.log_export_file.write('Generating kratos cases ...' + '\n')
        self.end_sim_file_num = 0
        # creat the cases_run.sh
        cases_run_path_and_name = os.path.join(os.getcwd(),'cases_run.sh')
        with open(cases_run_path_and_name, "w") as f_w_cases_run:
            f_w_cases_run.write('#!/bin/bash'+'\n')

            #loop every individual in the pop
            for indiv_ in nextoff:

                #Young_mudulus_particle = str(indiv_['Gene'].data[0])
                #Young_mudulus_bond     = str(indiv_['Gene'].data[1])
                #sigma_max_bond         = str(indiv_['Gene'].data[2])
                #cohesion_ini_bond      = str(indiv_['Gene'].data[3])
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
                            new_folder_name = 'G' + str(g_count) + '_' + confining_pressure + '_' + texture_angle + '_' \
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
                        
                        else: #confining pressure > 0.0

                            #creat new folder
                            new_folder_name = 'G' + str(g_count) + '_' + confining_pressure + '_' + texture_angle + '_' \
                                                + strong_p_E + '_' + strong_b_E + '_' + strong_b_knks + '_'\
                                                + weak_p_E + '_' + weak_b_E + '_' + weak_b_knks + '_'\
                                                + strong_b_n_max + '_' + strong_b_t_max + '_' + strong_b_phi + '_'\
                                                + weak_b_n_max + '_' + weak_b_t_max + '_' + weak_b_phi
                            aim_path = os.path.join(os.getcwd(),'Generated_kratos_cases', new_folder_name)
                            aim_path_change_marker = 0
                            aim_path, aim_path_change_marker = self.uniquify(aim_path, aim_path_change_marker)
                            os.mkdir(aim_path)

                            #copy source file
                            texture_angle_folder = 'angel_' + str(texture_angle)
                            seed_file_name_list = ['decompressed_material_triaxial_test_PBM_GA_230315.py', 'G-TriaxialDEM_FEM_boundary.mdpa',\
                                                    'G-TriaxialDEM.mdpa', 'ProjectParametersDEM.json', 'MaterialsDEM.json', 'run_omp.sh']
                            for seed_file_name in seed_file_name_list:
                                seed_file_path_and_name = os.path.join(os.getcwd(),'kratos_seed_files','Triaxial', texture_angle_folder, seed_file_name)
                                aim_file_path_and_name = os.path.join(aim_path, seed_file_name)

                                if seed_file_name == 'MaterialsDEM.json':
                                    with open(seed_file_path_and_name, "r") as f_material:
                                        with open(aim_file_path_and_name, "w") as f_material_w:
                                            for line in f_material.readlines():
                                                if "YOUNG_MODULUS" in line:
                                                    line = line.replace("5.0e10", str(Young_mudulus_particle))
                                                if "BOND_YOUNG_MODULUS" in line:
                                                    line = line.replace("3.0e8", str(Young_mudulus_bond))
                                                if "BOND_SIGMA_MAX" in line:
                                                    line = line.replace("1e5", str(sigma_max_bond))
                                                if "BOND_TAU_ZERO" in line:
                                                    line = line.replace("5e5", str(cohesion_ini_bond))
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
                    f_w_cases_run.write('cd '+ aim_path + '\n')
                    f_w_cases_run.write('sbatch run_omp.sh' + '\n')
                    self.end_sim_file_num += 1

    def run_kratos_cases(self):
        self.log_export_file.write('Running kratos cases ...' + '\n')
        command_execution = 'sh cases_run.sh'
        os.system(command_execution)
    
    def read_kratos_results_and_add_fitness(self, g_count, nextoff):
        
        self.log_export_file.write('Reading kratos results and adding fitness ...' + '\n')
        for indiv_ in nextoff:

            Young_mudulus_particle = str(indiv_['Gene'].data[0])
            Young_mudulus_bond     = str(indiv_['Gene'].data[1])
            sigma_max_bond         = str(indiv_['Gene'].data[2])
            cohesion_ini_bond      = str(indiv_['Gene'].data[3])

            #the strength files
            aim_folder_name = 'G' + str(g_count) + '_Ep' + Young_mudulus_particle + '_Eb' + Young_mudulus_bond\
                            + '_Sig' + sigma_max_bond + '_Coh' + cohesion_ini_bond
            aim_path_and_name = os.path.join(os.getcwd(),'Generated_kratos_cases', aim_folder_name, 'G-Triaxial_Graphs', 'G-Triaxial_graph.grf')

            if os.path.getsize(aim_path_and_name) != 0:
                stress_data_list = []
                strain_data_list = []
                with open(aim_path_and_name, 'r') as stress_strain_data:
                    for line in stress_strain_data:
                        values = [float(s) for s in line.split()]
                        stress_data_list.append(values[1]) 
                        strain_data_list.append(values[0])
                strength_max = max(stress_data_list)
                strain_max = strain_data_list[stress_data_list.index(max(stress_data_list))]
                rel_error_strength = ((strength_max - self.aim_strength) / self.aim_strength)**2
                rel_error_starin   = ((strain_max - self.aim_strain) / self.aim_strain)**2
            else:
                strength_max = 0.0
                strain_max = 0.0
                rel_error_strength = (self.aim_strength)**2
                rel_error_starin   = (self.aim_strain)**2

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
                if max(strain_data_list) > 0.6:
                    young_cnt = 0
                    young_select_sum = 0.0
                    for strain_data in strain_data_list:
                        if strain_data > 0.4 and strain_data < 0.6:
                            young_select_sum += young_data_list[strain_data_list.index(strain_data)]
                            young_cnt += 1
                    young_modulus_max = young_select_sum / young_cnt
                else:
                    young_modulus_max = max(young_data_list)
                rel_error_young_modulus = ((young_modulus_max - self.aim_young_modulus) / self.aim_young_modulus)**2
            else:
                young_modulus_max = 0.0
                rel_error_young_modulus = (self.aim_young_modulus)**2

            if rel_error_strength + rel_error_young_modulus + rel_error_starin:
                fitness = 1 / (rel_error_strength + rel_error_young_modulus + rel_error_starin)
            else:
                fitness = 0

            indiv_['fitness'] = fitness

            #write out individual data for ML
            output_file_name = 'G_info.csv' 
            aim_path_and_name = os.path.join(os.getcwd(),'kratos_results_data', output_file_name)

            indiv_data_head = ['Young_mudulus_particle', 'Young_mudulus_bond', 'sigma_max_bond', 'cohesion_ini_bond', 'strength_max', 'strain_max', 'young_modulus_max']
            indiv_data = []
            with open(aim_path_and_name, "a+", encoding='UTF8', newline='') as f_w:
                writer = csv.writer(f_w)
                indiv_data.append(indiv_['Gene'].data[0])
                indiv_data.append(indiv_['Gene'].data[1])
                indiv_data.append(indiv_['Gene'].data[2])
                indiv_data.append(indiv_['Gene'].data[3])
                indiv_data.append(strength_max)
                indiv_data.append(strain_max)
                indiv_data.append(young_modulus_max)
                if self.indiv_data_head_not_written:
                    writer.writerow(indiv_data_head)
                    self.indiv_data_head_not_written = False
                writer.writerow(indiv_data)
                f_w.close()

        return nextoff
    
    def save_and_plot_best_individual_results(self, g_count, best_individual):
        
        self.log_export_file.write('Saving and ploting best individual results ...' + '\n')
        #save data files
        new_file_name = 'best_individual_data.txt'
        aim_path_and_name = os.path.join(os.getcwd(),'kratos_results_data', new_file_name)

        with open(aim_path_and_name, "a+") as f_w:
            f_w.write(str(g_count) + ' ' + str(best_individual['Gene'].data[0])+ ' ' + str(best_individual['Gene'].data[1])+ ' ' + str(best_individual['Gene'].data[2])\
                      + ' ' + str(best_individual['Gene'].data[3]) + '\n')
        f_w.close()
        
        #plot and save
        Young_mudulus_particle = str(best_individual['Gene'].data[0])
        Young_mudulus_bond     = str(best_individual['Gene'].data[1])
        sigma_max_bond         = str(best_individual['Gene'].data[2])
        cohesion_ini_bond      = str(best_individual['Gene'].data[3])

        from_folder_name = 'G' + str(g_count) + '_Ep' + Young_mudulus_particle + '_Eb' + Young_mudulus_bond\
                        + '_Sig' + sigma_max_bond + '_Coh' + cohesion_ini_bond
        to_folder_name = 'G_' + str(g_count)
        self.log_export_file.write('Coping ' + from_folder_name + '\n')
        #save the best individual case to results folder
        from_directory = os.path.join(os.getcwd(),'Generated_kratos_cases', from_folder_name)
        to_directory = os.path.join(os.getcwd(),'kratos_results_data', to_folder_name)
        copy_tree(from_directory, to_directory)
    
    def final_clear_kratos_case_and_data_folder(self):

        kratos_case_folder_name = 'Generated_kratos_cases'
        if os.path.exists(kratos_case_folder_name):
            shutil.rmtree(kratos_case_folder_name, ignore_errors=True)

        kratos_temp_data_folder_name = 'kratos_results_data_temp'
        if os.path.exists(kratos_temp_data_folder_name):
            shutil.rmtree(kratos_temp_data_folder_name, ignore_errors=True)

        cases_run_path_and_name = os.path.join(os.getcwd(),'cases_run.sh')
        if os.path.exists(cases_run_path_and_name):
            os.remove(cases_run_path_and_name)
    
    def GA_main(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3]

        self.log_export_file = self.clear_old_and_creat_new_log_file()
 
        self.log_export_file.write("Start of evolution" + '\n')
        self.log_export_file.flush()

        start_time = time.time()
 
        # Begin the evolution
        for g in range(NGEN):
 
            #clear old kratos case files and creat new one
            self.clear_old_and_creat_new_kratos_case_folder()
            
            self.log_export_file.write("############### Generation {} ###############".format(g) + '\n')
 
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
                
                #the predicted best individual by inside GA are added to the population
                newoff1_add = Gene(data=[])  # offspring1 produced by cross operation
                newoff2_add = Gene(data=[])  # offspring2 produced by cross operation
                newoff1_add.data = self.bestindividual_in['Gene'].data
                newoff2_add.data = self.best_ind_in['Gene'].data
                nextoff.append({'Gene': newoff1_add})
                nextoff.append({'Gene': newoff2_add})
                popsize += 2
                
            else:
                nextoff = self.pop

            #generate kratos cases according to pop 
            self.generate_kratos_cases(g, nextoff)

            self.run_kratos_cases()

            self.log_export_file.flush()

            #check whether all the kratos cases in this generation finished
            file_num = 0
            time_count = 0
            while file_num != self.end_sim_file_num:
                aim_path_and_folder = os.path.join(os.getcwd(),'kratos_results_data_temp')
                file_num = len(glob.glob1(aim_path_and_folder,"*.txt"))
                time.sleep(30)
                self.log_export_file.write('-----Waiting for kratos cases -----' + '\n')
                time_count += 0.5
                self.log_export_file.write('-------Generation {} cost {} min(s)-------'.format(g, time_count) + '\n')
                self.log_export_file.flush()

            #add fitness to nextoff
            nextoff = self.read_kratos_results_and_add_fitness(g, nextoff)

            # The population is entirely replaced by the offspring
            self.pop = nextoff
 
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
 
            best_ind = self.selectBest(self.pop)
 
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
                # save the data of the best individual for post processing
                self.save_and_plot_best_individual_results(g, self.bestindividual)
                self.log_export_file.write('Saving best_individual' + '\n')
            else:
                self.save_and_plot_best_individual_results(g, best_ind)
                self.log_export_file.write('Saving best_ind' + '\n')
 
            self.log_export_file.write("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']) + '\n')
            self.log_export_file.write(" Max fitness of current pop: {}".format(max(fits)) + '\n')

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_export_file.write('Total simulation time cost is {}'.format(elapsed_time) + '\n')
            self.log_export_file.flush()

            ############# ML part################
            data_min_list = self.parameter[4]
            data_max_list = self.parameter[5]

            #strength predictor
            predict_index = 4
            run_ml_4 = MachineLearning()
            ML_xgb_4 = run_ml_4.ML_main(data_min_list, data_max_list, predict_index)

            #strength predictor
            predict_index = 5
            run_ml_5 = MachineLearning()
            ML_xgb_5 = run_ml_5.ML_main(data_min_list, data_max_list, predict_index)

            #strength predictor
            predict_index = 6
            run_ml_6 = MachineLearning()
            ML_xgb_6 = run_ml_6.ML_main(data_min_list, data_max_list, predict_index)

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
                            fit_muteoff1_in = self.evaluate_in(muteoff1_in.data, ML_xgb_4, run_ml_4, ML_xgb_5, ML_xgb_6)  # Evaluate the individuals
                            fit_muteoff2_in = self.evaluate_in(muteoff2_in.data, ML_xgb_4, run_ml_4, ML_xgb_5, ML_xgb_6)  # Evaluate the individuals
                            #fit_muteoff1_in = 0.1
                            #fit_muteoff2_in = 0.01
                            nextoff_in.append({'Gene': muteoff1_in, 'fitness': fit_muteoff1_in})
                            nextoff_in.append({'Gene': muteoff2_in, 'fitness': fit_muteoff2_in})
                        else:
                            fit_crossoff1_in = self.evaluate_in(crossoff1_in.data, ML_xgb_4, run_ml_4, ML_xgb_5, ML_xgb_6)  # Evaluate the individuals
                            fit_crossoff2_in = self.evaluate_in(crossoff2_in.data, ML_xgb_4, run_ml_4, ML_xgb_5, ML_xgb_6)
                            #fit_crossoff1_in = 0.1
                            #fit_crossoff2_in = 0.01
                            nextoff_in.append({'Gene': crossoff1_in, 'fitness': fit_crossoff1_in})
                            nextoff_in.append({'Gene': crossoff2_in, 'fitness': fit_crossoff2_in})
                    else:
                        nextoff_in.extend(offspring_in)

                # The population is entirely replaced by the offspring
                self.pop_in = nextoff_in
    
                self.best_ind_in = self.selectBest(self.pop_in)
    
                if self.best_ind_in['fitness'] > self.bestindividual_in['fitness']:
                    self.bestindividual_in = self.best_ind_in

            self.log_export_file.write("Best individual in inside GA found is {}, {}".format(self.bestindividual_in['Gene'].data,
                                                                        self.bestindividual_in['fitness']) + '\n')
            self.log_export_file.flush()
            
        self.final_clear_kratos_case_and_data_folder()
        self.log_export_file.write("------ End of (successful) evolution ------" + '\n')
        self.log_export_file.flush()
        self.log_export_file.close()
 
if __name__ == "__main__":
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.2, 500, 200  # popsize must be even number
    aim_strength, aim_young_modulus = 4.323e7, 5.54e9
    aim_strain = 1.01265
 
    #variables list[strong_p_E, strong_b_E, strong_b_knks, weak_p_E, weak_b_E, weak_b_knks,
    #               strong_b_n_max, strong_b_t_max, strong_b_phi, weak_b_n_max, weak_b_t_max, weak_b_phi]
    low = [5e8, 5e8, 1, 5e8, 5e8, 1, 2e6, 2e6, 0, 2e6, 2e6, 0]  # lower range for variables
    up  = [1e11, 1e11, 3, 1e11, 1e11, 3, 2e8, 2e8, 50, 2e8, 2e8, 50]  # upper range for variables
    parameter = [CXPB, MUTPB, NGEN, popsize, low, up, aim_strength, aim_young_modulus, aim_strain]
    run = GA(parameter)
    run.GA_main()