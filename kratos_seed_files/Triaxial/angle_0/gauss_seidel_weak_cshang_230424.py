# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_coupled_solver import CoSimulationCoupledSolver

def Create(settings, models, solver_name):
    return GaussSeidelWeakCoupledSolver(settings, models, solver_name)

class GaussSeidelWeakCoupledSolver(CoSimulationCoupledSolver):
    def SolveSolutionStep(self):
        for coupling_op in self.coupling_operations_dict.values():
            coupling_op.InitializeCouplingIteration()

        for solver_name, solver in self.solver_wrappers.items():
            self._SynchronizeInputData(solver_name)
            solver.SolveSolutionStep()
            self._SynchronizeOutputData(solver_name)

        for coupling_op in self.coupling_operations_dict.values():
            coupling_op.FinalizeCouplingIteration()

        return True
    
    def GetStrain(self):
        for solver_name, solver in self.solver_wrappers.items():
            if solver_name == 'dem':
                return solver.GetStrain()
            
    def GetTotalStressMeanMaxTime(self):
        for solver_name, solver in self.solver_wrappers.items():
            if solver_name == 'dem':
                return solver.GetTotalStressMeanMaxTime()