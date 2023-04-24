import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis

"""
For user-scripting it is intended that a new class is derived
from CoSimulationAnalysis to do modifications
Check also "kratos/python_scripts/analysis_stage.py" for available methods that can be overridden
"""

class AutoEndCoSimulationAnalysis(CoSimulationAnalysis):

    def __init__(self, parameters, models=None):
        super().__init__(parameters)
        self.end_sim = 0

    def RunSolutionLoop(self):

        while self.KeepAdvancingSolutionLoop():
            self.time = self._GetSolver().AdvanceInTime(self.time)
            self.InitializeSolutionStep()
            self._GetSolver().Predict()
            self._GetSolver().SolveSolutionStep()
            self.FinalizeSolutionStep()
            self.OutputSolutionStep()
    
    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        self.CheckSimulationEnd()

    def KeepAdvancingSolutionLoop(self):
        
        return (self.time < self.end_time and self.end_sim < 1)
    
    def CheckSimulationEnd(self):
       
        total_stress_mean_max_time = self._GetSolver().GetTotalStressMeanMaxTime()
        strain = self._GetSolver().GetStrain()

        if total_stress_mean_max_time < 0.5 * self.time or strain > 2.0:
            self.end_sim = 2   # means end the simulation

parameter_file_name = "ProjectParametersCoSim.json"
with open(parameter_file_name,'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

simulation = AutoEndCoSimulationAnalysis(parameters)
simulation.Run()
