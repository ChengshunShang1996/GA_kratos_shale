#/////////////////////////////////////////////////
#// Author: Chengshun Shang (CIMNE)
#// Email: cshang@cimne.upc.edu
#// Date: Mar 2023
#/////////////////////////////////////////////////

# Importing the Kratos Library
import KratosMultiphysics as KM

# Importing the Kratos Library
from KratosMultiphysics.kratos_utilities import CheckIfApplicationsAvailable

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.solver_wrappers.kratos import kratos_base_wrapper

# Importing StructuralMechanics
if not CheckIfApplicationsAvailable("DEMApplication"):
    raise ImportError("The DEMApplication is not available!")

from KratosMultiphysics import DEMApplication
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

#for dem_wrapper_cshang
from KratosMultiphysics import *
from KratosMultiphysics.DEMApplication import *
import os
import math
import datetime

def Create(settings, model, solver_name):
    return DEMWrapper(settings, model, solver_name)

class DEMWrapper(kratos_base_wrapper.KratosBaseWrapper):
    """This class is the interface to the DEMApplication of Kratos"""

    def _CreateAnalysisStage(self):
            
        class CoTriaxialTest(DEMAnalysisStage):
            """This derived class mainly used for the stress-strain data output"""

            def __init__(self, model, parameters):
                super().__init__(model, parameters)
                self.parameters = parameters
                self.end_sim = 0
            
            def Initialize(self):
                super().Initialize()
                self.InitializeMaterialTest()
                self.PrepareDataForGraph()

            def RunSolutionLoop(self):

                print("\n************** Applying standard triaxial...\n", flush=True)
                while self.KeepAdvancingSolutionLoop():
                    self.time = self._GetSolver().AdvanceInTime(self.time)
                    self.InitializeSolutionStep()
                    self._GetSolver().Predict()
                    self._GetSolver().SolveSolutionStep()
                    self.FinalizeSolutionStep()
                    self.OutputSolutionStep()
                print("\n************** Finished Applying standard triaxial...\n", flush=True)

            def OutputSolutionStep(self):
                super().OutputSolutionStep()
                self.PrintGraph(self.time)

            def FinalizeSolutionStep(self):
                super().FinalizeSolutionStep()
                self.MeasureForcesAndPressure()
                self.CheckSimulationEnd()

            def Finalize(self):
                super().Finalize()
                self.FinalizeGraphs()
                self.CreatEndMarkerFile()

            def InitializeMaterialTest(self):

                self.top_mesh_nodes = []
                self.bot_mesh_nodes = []
                self.graph_counter = 0
                self.CN_graph_counter = 0
                self.length_correction_factor = 1.0
                self.graph_frequency  = int(self.parameters["GraphExportFreq"].GetDouble()/self.parameters["MaxTimeStep"].GetDouble())
                self.strain = 0.0 
                self.total_stress_top = 0.0; self.total_stress_bot = 0.0; self.total_stress_mean = 0.0
                self.LoadingVelocity = 0.0
                self.MeasuringSurface = 0.0
                self.total_stress_mean_max = 0.0
                self.total_stress_mean_max_time = 0.0
                self.young_modulus = 0.0
                self.young_cal_counter = 0
                self.last_stress_mean_for_young_cal = 0.0
                self.last_strain_mean_for_young_cal = 0.0

                if "material_test_settings" in self.parameters.keys():
                    self.height = self.parameters["material_test_settings"]["SpecimenLength"].GetDouble()
                    self.diameter = self.parameters["material_test_settings"]["SpecimenDiameter"].GetDouble()
                    self.test_type = self.parameters["material_test_settings"]["TestType"].GetString()
                    self.y_coordinate_of_cylinder_bottom_base = self.parameters["material_test_settings"]["YCoordinateOfCylinderBottomBase"].GetDouble()
                    self.z_coordinate_of_cylinder_bottom_base = self.parameters["material_test_settings"]["ZCoordinateOfCylinderBottomBase"].GetDouble()
                else:
                    self.height = self.parameters["SpecimenLength"].GetDouble()
                    self.diameter = self.parameters["SpecimenDiameter"].GetDouble()
                    self.test_type = self.parameters["TestType"].GetString()
                    self.y_coordinate_of_cylinder_bottom_base = self.parameters["YCoordinateOfCylinderBottomBase"].GetDouble()
                    self.z_coordinate_of_cylinder_bottom_base = self.parameters["ZCoordinateOfCylinderBottomBase"].GetDouble()

                self.ComputeLoadingVelocity()
                self.ComputeMeasuringSurface()
                self.problem_name = self.parameters["problem_name"].GetString()
                self.initial_time = datetime.datetime.now()
                absolute_path_to_file = os.path.join(self.graphs_path, self.problem_name + "_Parameter_chart.grf")
                self.chart = open(absolute_path_to_file, 'w')
                self.aux = AuxiliaryUtilities()
                self.PreUtilities = PreUtilities()
                self.PrepareTests()

                if self.parameters["PostGroupId"].GetBool() is True:
                    self.SetGroupIDtoParticle()

            def KeepAdvancingSolutionLoop(self):
        
                return (self.time < self.end_time and self.end_sim < 1)

            def ComputeLoadingVelocity(self):
                top_vel = bot_vel = 0.0
                for smp in self.rigid_face_model_part.SubModelParts:
                    if smp[IDENTIFIER] == 'TOP':
                        top_vel = smp[LINEAR_VELOCITY_Y]
                    if smp[IDENTIFIER] == 'BOTTOM':
                        bot_vel = smp[LINEAR_VELOCITY_Y]
                self.LoadingVelocity = top_vel - bot_vel

            def ComputeMeasuringSurface(self):
                self.MeasuringSurface = 0.25 * math.pi * self.diameter * self.diameter

            def PrepareTests(self):

                absolute_path_to_file1 = os.path.join(self.graphs_path, self.problem_name + "_graph.grf")
                absolute_path_to_file2 = os.path.join(self.graphs_path, self.problem_name + "_graph_top.grf")
                absolute_path_to_file3 = os.path.join(self.graphs_path, self.problem_name + "_graph_bot.grf")
                absolute_path_to_file4 = os.path.join(self.graphs_path, self.problem_name + "_graph_young.grf")
                self.graph_export_1 = open(absolute_path_to_file1, 'w')
                self.graph_export_2 = open(absolute_path_to_file2, 'w')
                self.graph_export_3 = open(absolute_path_to_file3, 'w')
                self.graph_export_4 = open(absolute_path_to_file4, 'w')

                self.procedures.KratosPrintInfo('Initial Height of the Model: ' + str(self.height)+'\n')

                absolute_path_to_file = os.path.join(self.graphs_path, self.problem_name + "_CN.grf")
                self.CN_export = open(absolute_path_to_file, 'w')

            def PrepareDataForGraph(self):

                prepare_check = [0,0,0,0]
                self.total_check = 0

                for smp in self.rigid_face_model_part.SubModelParts:
                    if smp[IDENTIFIER] == 'TOP':
                        self.top_mesh_nodes = smp.Nodes
                        prepare_check[0] = 1
                    if smp[IDENTIFIER] == 'BOTTOM':
                        self.bot_mesh_nodes = smp.Nodes
                        prepare_check[1] = 1

                for smp in self.spheres_model_part.SubModelParts:
                    if smp[IDENTIFIER] == 'TOP':
                        self.top_mesh_nodes = smp.Nodes
                        prepare_check[2] = -1

                    if smp[IDENTIFIER] == 'BOTTOM':
                        self.bot_mesh_nodes = smp.Nodes
                        prepare_check[3] = -1

                for it in range(len(prepare_check)):
                    self.total_check += prepare_check[it]

                if math.fabs(self.total_check) != 2:
                    self.Procedures.KratosPrintWarning(" ERROR in the definition of TOP BOT groups. Both groups are required to be defined, they have to be either on FEM groups or in DEM groups")

            def SetGroupIDtoParticle(self):
        
                #for shale rock simulation
                self.joint_model_part = self.spheres_model_part.GetSubModelPart('DEMParts_Joint')

                for element in self.joint_model_part.Elements:
                    element.GetNode(0).SetSolutionStepValue(GROUP_ID, 1)

                self.body_model_part = self.spheres_model_part.GetSubModelPart('DEMParts_Body')

                for element in self.body_model_part.Elements:
                    element.GetNode(0).SetSolutionStepValue(GROUP_ID, 0)
            
            def MeasureForcesAndPressure(self):

                dt = self.parameters["MaxTimeStep"].GetDouble()
                strain_delta = -100 * self.length_correction_factor * self.LoadingVelocity * dt / self.height
                self.strain += strain_delta

                total_force_top = 0.0
                for node in self.top_mesh_nodes:
                    force_node_y = node.GetSolutionStepValue(ELASTIC_FORCES)[1]
                    total_force_top += force_node_y
                self.total_stress_top = total_force_top / self.MeasuringSurface

                total_force_bot = 0.0
                for node in self.bot_mesh_nodes:
                    force_node_y = -node.GetSolutionStepValue(ELASTIC_FORCES)[1]
                    total_force_bot += force_node_y
                self.total_stress_bot = total_force_bot / self.MeasuringSurface

                self.total_stress_mean = 0.5 * (self.total_stress_bot + self.total_stress_top)

                if self.young_cal_counter == self.graph_frequency:
                    self.young_cal_counter = 0
                    stress_mean_delta = self.total_stress_mean - self.last_stress_mean_for_young_cal
                    strain_delta_for_young_cal = self.strain - self.last_strain_mean_for_young_cal
                    if strain_delta_for_young_cal != 0.0:
                        self.young_modulus = stress_mean_delta / (strain_delta_for_young_cal / 100)
                    else:
                        print("*************************strain_delta_for_young_cal is 0.0 !***************")
                    self.last_stress_mean_for_young_cal = self.total_stress_mean
                    self.last_strain_mean_for_young_cal = self.strain
                self.young_cal_counter += 1

                if self.total_stress_mean_max < self.total_stress_mean:
                    self.total_stress_mean_max = self.total_stress_mean
                    self.total_stress_mean_max_time = self.time

            def CheckSimulationEnd(self):

                #if self.total_stress_mean_max_time < 0.5 * self.time or self.strain > 5.0:
                if self.total_stress_mean_max_time < 0.5 * self.time or self.strain > 8.0:
                    self.end_sim = 2   # means end the simulation
                    
            def PrintGraph(self, time):

                if self.graph_counter == self.graph_frequency:
                    self.graph_counter = 0
                    self.graph_export_1.write(str("%.6g"%self.strain).rjust(13) + "  " + str("%.6g"%(self.total_stress_mean)).rjust(13) + "  " + str("%.8g"%time).rjust(12) + '\n')
                    self.graph_export_2.write(str("%.8g"%self.strain).rjust(13) + "  " + str("%.6g"%(self.total_stress_top)).rjust(13) + "  " + str("%.8g"%time).rjust(12) + '\n')
                    self.graph_export_3.write(str("%.8g"%self.strain).rjust(13) + "  " + str("%.6g"%(self.total_stress_bot)).rjust(13) + "  " + str("%.8g"%time).rjust(12) + '\n')
                    self.graph_export_4.write(str("%.8g"%self.strain).rjust(13) + "  " + str("%.6g"%(self.young_modulus)).rjust(13) + "  " + str("%.8g"%time).rjust(12) + '\n')
                    self.graph_export_1.flush()
                    self.graph_export_2.flush()
                    self.graph_export_3.flush()
                    self.graph_export_4.flush()
                self.graph_counter += 1
            
            def FinalizeGraphs(self):
                self.graph_export_1.close()
                self.graph_export_2.close()
                self.graph_export_3.close()
                self.graph_export_4.close()

            def uniquify(self, path):

                filename, extension = os.path.splitext(path)
                counter = 1

                while os.path.exists(path):
                    path = filename + "_" + str(counter) + extension
                    counter += 1

                return path
            
            def CreatEndMarkerFile(self):

                current_folder_name = os.path.basename(os.path.dirname(__file__))
                two_up_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

                aim_file_name = current_folder_name + '.txt'
                aim_path = os.path.join(two_up_path,'kratos_results_data_temp', aim_file_name)

                aim_path = self.uniquify(aim_path)

                with open(aim_path, "w") as marker_file:
                    marker_file.write('HOLA Barcelona!'+'\n')
                marker_file.close()

        dem_analysis_module = CoTriaxialTest

        return dem_analysis_module(self.model, self.project_parameters)

    def Initialize(self):
        super().Initialize()

        # save nodes in model parts which need to be moved while simulating
        self.list_of_nodes_in_move_mesh_model_parts = [self.model[mp_name].Nodes for mp_name in self.settings["solver_wrapper_settings"]["move_mesh_model_part"].GetStringArray()]


    def SolveSolutionStep(self):
        # move the rigid wall object in the dem mp w.r.t. the current displacement and velocities
        for model_part_nodes in self.list_of_nodes_in_move_mesh_model_parts:
            DEMApplication.MoveMeshUtility().MoveDemMesh(model_part_nodes,True)
        # solve DEM
        super().SolveSolutionStep()

