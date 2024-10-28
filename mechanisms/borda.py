from scipy.stats import genlogistic

from config import config
from markov_decision_processes.gurobi_helper import init_gurobi_model
from markov_decision_processes.occupancy_measure import add_occupancy_polytope_constraints
from mechanisms.aggregation_mechanism import AggregationRule

from gurobipy import GRB
import gurobipy as gp
import numpy as np


class BordaRule(AggregationRule):
    def __init__(self, mdp, agents):
        super().__init__(mdp, agents)
        self.name = 'borda'

    def check_feasibility_concave_portion(self, log_verbose=False):
        model = init_gurobi_model(name=f'borda rule')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)
        for i, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            required_utility = ag.gen_log_mode
            print(f"\t\t\t\tBorda agent {ag.id} requires utility {required_utility}")
            model.addConstr(
                ag_return >= config.SCALING_D_PI * required_utility,
                name=f'agent_{i}_borda_approve'
            )

        model.optimize()
        print("Status Borda feasibility", model.status)
        if model.status != GRB.OPTIMAL:
            return False
        return True

    def solve(self, log_verbose=False):
        if not self.check_feasibility_concave_portion(log_verbose=log_verbose):
            raise RuntimeError(f"Borda in the concave portion is infeasible.")
        print("Borda concave portion is feasible")
        model = init_gurobi_model(name=f'borda rule')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)
        f = model.addVars(self.n_agents, vtype=GRB.CONTINUOUS, name='borda_scores')
        model.setObjective(gp.quicksum(f), GRB.MAXIMIZE)

        for i, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            required_utility = ag.gen_log_mode
            step_size = 0.01

            def get_line(_x):
                a = genlogistic.pdf(_x, *ag.gen_log_params)
                y = genlogistic.cdf(_x, *ag.gen_log_params)
                b = y - a * _x
                return a, b

            ag_lines = list()
            for x in np.arange(required_utility, ag.max_expected_reward, step_size):
                l_x = get_line(x)
                if len(ag_lines) == 0 or ag_lines[-1][0] - 0.001 >= l_x[0]:
                    ag_lines.append(l_x)
            for _ind, ag_l in enumerate(ag_lines):
                model.addConstr(
                    f[i] <= 100 * (ag_return / config.SCALING_D_PI * ag_l[0] + ag_l[1]),
                    name=f'agent_{_ind}_borda_line'
                )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Borda rule failed. Status: {model.status}.")
        max_borda = model.getObjective().getValue()
        occupancy_measure = np.array([
            [d_pi[s, a].x / config.SCALING_D_PI for a in range(self.mdp.n_actions)] for s in range(self.mdp.n_states)
        ])
        if log_verbose:
            print(f"Computed the max borda score: {max_borda}")
            print(f"\tAgents borda scores: {[f[i].x for i in range(len(self.agents))]}")
            for i, a in enumerate(self.agents):
                print(f"\t\tAgent {i} gets reward {(occupancy_measure * a.reward).sum()} "
                      f"(max reward: {a.max_expected_reward}, min reward: {a.min_expected_reward})")
        info = {
            'borda_scores': [f[i].x for i in range(len(self.agents))]
        }
        return info, occupancy_measure
