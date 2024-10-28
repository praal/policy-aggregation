from config import config
from markov_decision_processes.gurobi_helper import init_gurobi_model
from markov_decision_processes.occupancy_measure import add_occupancy_polytope_constraints
from mechanisms.aggregation_mechanism import AggregationRule

from gurobipy import GRB
import gurobipy as gp
import numpy as np


class MaxQuantileRule(AggregationRule):
    def __init__(self, mdp, agents):
        super().__init__(mdp, agents)
        self.name = 'quantile'

    def check_quantile_feasibility(self, alpha, completion_method=None, log_verbose=False):
        model = init_gurobi_model(name=f'max quantile')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)

        if completion_method == 'utilitarian':
            model.setObjective(gp.quicksum(
                    gp.quicksum(ag.reward[s, a] for ag in self.agents) * d_pi[s, a]
                    for a in range(self.mdp.n_actions)
                    for s in range(self.mdp.n_states)
            ), GRB.MAXIMIZE)
        elif completion_method == 'egalitarian':
            delta = model.addVar(vtype=GRB.CONTINUOUS)
            model.setObjective(delta, GRB.MAXIMIZE)
        elif completion_method is None:
            pass
        else:
            raise NotImplementedError()

        for ind, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            required_utility = ag.alpha_quantile_utility(alpha)
            model.addConstr(
                ag_return >= config.SCALING_D_PI * required_utility,
                name=f'agent_{ind}_approval_count'
            )
            if completion_method == 'egalitarian':
                model.addConstr(
                    ag_return >= delta,
                    name=f'agent_{ind}_egal_utility'
                )

        model.optimize()

        if model.status == GRB.INFEASIBLE:
            return None, None
        elif model.status != GRB.OPTIMAL:
            raise RuntimeError(f"max-quantile with completion {completion_method} rule failed for {alpha}."
                               f"Status: {model.status}.")
        max_welfare = model.getObjective().getValue() / config.SCALING_D_PI
        occupancy_measure = np.array([
            [d_pi[s, a].x / config.SCALING_D_PI for a in range(self.mdp.n_actions)] for s in range(self.mdp.n_states)
        ])
        if log_verbose:
            for i, a in enumerate(self.agents):
                print(f"\t\tAgent {i} gets reward {(occupancy_measure * a.reward).sum()} "
                      f"(max reward: {a.max_expected_reward}, min reward: {a.min_expected_reward})")
        return max_welfare, occupancy_measure

    def solve(self, completion_method=None, log_verbose=False):
        l_sampled_policies = len(self.agents[0].histogram_policy_returns)
        st, en = 0, l_sampled_policies + 1
        while en > st + 1:
            print("binary search max quantile:", st, en)
            mid = (en + st) // 2
            max_welfare, occupancy_measure = self.check_quantile_feasibility(mid / l_sampled_policies)
            if max_welfare is None:
                en = mid
            else:
                st = mid
        q = st / l_sampled_policies
        max_welfare, occupancy_measure = self.check_quantile_feasibility(
            q, completion_method=completion_method, log_verbose=log_verbose)
        if log_verbose:
            print(f"Max feasible quantile was: {q}, with completion: {completion_method}, welfare: {max_welfare}")
        info = {
            'max-q': q,
            'welfare': max_welfare
        }
        return info, occupancy_measure
