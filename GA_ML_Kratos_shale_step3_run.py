import os
import glob
import time
import math
 
class GA:
    """
    This is a class of GA algorithm.
    """
 
    def __init__(self):
        
        self.end_sim_file_num = 1800

    def run_kratos_cases(self):

        #submit 30 jobs at begining
        file_num = 0
        time_count = 0
        end_job_cnt = 0
        nodes_num = 16
        max_index = 20
        total_job_cnt = math.ceil(self.end_sim_file_num / nodes_num)
        if max_index > total_job_cnt:
            max_index = total_job_cnt

        for i in range(0,max_index):
            cases_run_name = 'cases_run_' + str(i) + '.sh'
            command_execution = 'sbatch kratos_results_data_temp/' + cases_run_name
            os.system(command_execution)
        
        submited_job_cnt = max_index
        
        while file_num != self.end_sim_file_num and submited_job_cnt < total_job_cnt:
            aim_path_and_folder = os.path.join(os.getcwd(),'kratos_results_data_temp')
            file_num = len(glob.glob1(aim_path_and_folder,"*.txt"))
            time.sleep(30)
            time_count += 0.5 

            new_end_job_cnt = file_num // nodes_num
            if new_end_job_cnt > end_job_cnt:
                new_add_end_cnt = new_end_job_cnt - end_job_cnt
                end_job_cnt = new_end_job_cnt
                if end_job_cnt != 0:
                    for j in range (0, new_add_end_cnt):
                        i += 1
                        cases_run_name = 'cases_run_' + str(i) + '.sh'
                        command_execution = 'sbatch kratos_results_data_temp/' + cases_run_name
                        if submited_job_cnt < total_job_cnt:
                            os.system(command_execution)
                        submited_job_cnt += 1
    
    def GA_main(self):
        """
        main frame work of GA
        """
        self.run_kratos_cases()

 
if __name__ == "__main__":
    run = GA()
    run.GA_main()