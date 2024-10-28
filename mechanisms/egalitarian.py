from config import config
from markov_decision_processes.gurobi_helper import init_gurobi_model
from markov_decision_processes.occupancy_measure import add_occupancy_polytope_constraints
from mechanisms.aggregation_mechanism import AggregationRule

from gurobipy import GRB
import gurobipy as gp
import numpy as np


class EgalitarianRule(AggregationRule):
    def __init__(self, mdp, agents):
        super().__init__(mdp, agents)
        self.name = 'egalitarian'

    def max_egal(self, fixed_utilities=None, log_verbose=False):
        if fixed_utilities is None:
            fixed_utilities = dict()
        model = init_gurobi_model(name=f'egal')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)
        delta = model.addVar(lb=-20000, ub=20000, vtype=GRB.CONTINUOUS)
        model.setObjective(delta, GRB.MAXIMIZE)
        cnt_non_fixed_agents = 0

        for i, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            if i in fixed_utilities:
                model.addConstr(
                    ag_return >= config.SCALING_D_PI * fixed_utilities[i] - 1e-4,
                    name=f'agent_{i}_fixed_utility'
                )
            else:
                model.addConstr(
                    # -1.0 * ag_return <= -1.0 * delta,
                    ag_return >= config.SCALING_D_PI * delta,
                    name=f'agent_{i}_at_least_delta'
                )
                cnt_non_fixed_agents += 1
        if not cnt_non_fixed_agents:
            raise RuntimeError("must have at least one non-fixed agent")

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Egal failed. Status: {model.status}.")
        max_egal = model.getObjective().getValue()
        min_u_agent = None
        for i, ag in enumerate(self.agents):
            if i in fixed_utilities:
                continue
            u_ag = np.sum(
                ag.reward[s, a] * d_pi[s, a].x / config.SCALING_D_PI
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            if min_u_agent is None or u_ag < min_u_agent[0]:
                min_u_agent = (u_ag, i)
        assert np.abs(min_u_agent[0] - max_egal) < 1e-4
        occupancy_measure = np.array([
            [d_pi[s, a].x / config.SCALING_D_PI for a in range(self.mdp.n_actions)] for s in range(self.mdp.n_states)
        ])
        new_utility_req = fixed_utilities.copy()
        new_utility_req[min_u_agent[1]] = min_u_agent[0]
        print(f"Egal, new agent: {min_u_agent[1]} requires {min_u_agent[0]}")
        return new_utility_req, occupancy_measure

    def solve(self, log_verbose=False):
        util_req = None
        for i in range(len(self.agents)):
            util_req, occupancy_measure = self.max_egal(util_req, log_verbose)
        info = {
            'utilities': util_req
        }
        return info, occupancy_measure
