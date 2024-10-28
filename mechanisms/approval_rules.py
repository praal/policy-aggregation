from config import config
from markov_decision_processes.gurobi_helper import init_gurobi_model
from markov_decision_processes.occupancy_measure import add_occupancy_polytope_constraints
from mechanisms.aggregation_mechanism import AggregationRule

from gurobipy import GRB
import gurobipy as gp
import numpy as np


class ApprovalsRule(AggregationRule):
    def __init__(self, mdp, agents, alpha=1):
        super().__init__(mdp, agents)
        self.alpha = alpha
        self.name = f"{alpha}-approvals"

    def solve_for_given_subset(self, must_approve, completion_method, log_verbose=False):
        assert len(must_approve) == self.n_agents
        approving_agents = [(i, ag) for i, ag in enumerate(self.agents) if must_approve[i]]

        model = init_gurobi_model(name=f'{self.alpha}-approval subset')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)

        if completion_method == 'utilitarian':
            model.setObjective(gp.quicksum(
                    gp.quicksum(ag.reward[s, a] for ag in self.agents) * d_pi[s, a]
                    for a in range(self.mdp.n_actions)
                    for s in range(self.mdp.n_states)
            ), GRB.MAXIMIZE)
        elif completion_method == 'egalitarian':
            delta = model.addVar(lb=-20000, ub=20000, vtype=GRB.CONTINUOUS)
            model.setObjective(delta, GRB.MAXIMIZE)
        else:
            raise NotImplementedError()

        for ind, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            if must_approve[ind]:
                required_utility = ag.alpha_quantile_utility(self.alpha)
                model.addConstr(
                    ag_return >= config.SCALING_D_PI * required_utility,
                    name=f'agent_{ind}_approval_count'
                )
            if completion_method == 'egalitarian':
                model.addConstr(
                    -1.0 * ag_return <= -delta,
                    name=f'agent_{ind}_approval_count'
                )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"{self.alpha}-approvals with completion {completion_method} rule"
                               f"for specific approvers of size {len(approving_agents)} failed."
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

    def find_max_approvals(self, log_verbose=False):
        model = init_gurobi_model(name=f'{self.alpha}-approval rule')
        d_pi = add_occupancy_polytope_constraints(model, self.mdp)
        approves = model.addVars(self.n_agents, vtype=GRB.BINARY, name='approves')
        model.setObjective(gp.quicksum(approves), GRB.MAXIMIZE)

        for i, ag in enumerate(self.agents):
            ag_return = gp.quicksum(
                ag.reward[s, a] * d_pi[s, a]
                for a in range(self.mdp.n_actions)
                for s in range(self.mdp.n_states)
            )
            required_utility = ag.alpha_quantile_utility(self.alpha)
            print(f"\t\t\t\t{self.alpha}-approval agent {i} requires utility {required_utility}")
            model.addConstr(
                (
                    max(1000, ag.max_expected_reward) * (1 - approves[i]) + ag_return / config.SCALING_D_PI
                    >= required_utility
                ),
                name=f'agent_{i}_approval_count'
            )

        if self.alpha == 1:
            model.write("model.lp")

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"{self.alpha}-approvals rule failed. Status: {model.status}.")
        max_approval_score = model.getObjective().getValue()
        occupancy_measure = np.array([
            [d_pi[s, a].x / config.SCALING_D_PI for a in range(self.mdp.n_actions)] for s in range(self.mdp.n_states)
        ])
        if log_verbose:
            print(f"Computed the max {self.alpha}-approval score: {max_approval_score}")
            print(f"\tAgents approving: {[approves[i].x for i in range(len(approves))]}")
            for i, a in enumerate(self.agents):
                print(f"\t\tAgent {i} gets reward {(occupancy_measure * a.reward).sum()} "
                      f"(max reward: {a.max_expected_reward}, min reward: {a.min_expected_reward})")
        return [int(approves[i].x > 0.5) for i in range(len(approves))], occupancy_measure

    def solve(self, log_verbose=False, **kwargs):
        approvals, occupancy_measure = self.find_max_approvals(log_verbose=log_verbose)
        backup_agents = self.agents
        if sum(approvals) == 0:
            print("Didn't find any, forcing one agent.")
            self.agents = [self.agents[0]]
            approvals, occupancy_measure = self.find_max_approvals(log_verbose=log_verbose)
            assert sum(approvals) == 1
        completion = kwargs.get('completion', 'utilitarian')
        ret = self.solve_for_given_subset(must_approve=approvals, completion_method=completion)
        self.agents = backup_agents
        info = {
            'approvals': approvals,
            'welfare': ret[0],
            'completion': completion
        }
        return info, ret[1]
