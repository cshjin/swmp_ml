#= Julia function wrapper to re-evaluate the optimal solution =#
using PowerModelsGMD

import SCIP
import Juniper
import JuMP
import PowerModels
import InfrastructureModels
import JSON
import Ipopt

const _PMGMD = PowerModelsGMD
const _IM = InfrastructureModels
const _PM = PowerModels

function re_eval(filename::String)
  """ Re-evaluate the model given the gic-blocker placement

  Returns:
    - result: a dictionary of the solution
  """
  # create a solver instance
  scip_solver = JuMP.optimizer_with_attributes(SCIP.Optimizer, "display/verblevel" => 0, "numerics/feastol" => 1e-4)

  # set solver options
  _PMGMD.silence()

  # read the case from the file
  case = _PM.parse_file(filename)

  # solve the model
  result = _PMGMD.solve_ac_gmd_mld(case, scip_solver)
  # println(result["termination_status"])
  # println(result["primal_status"])


  # return a tuple for feasibility and optimality
  # return result["primal_status"], result["objective"]
  return result["termination_status"], result["objective"], result
end