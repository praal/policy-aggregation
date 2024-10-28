from config import config
from markov_decision_processes.gurobi_helper import init_gurobi_model
from markov_decision_processes.markov_decision_process import MarkovDecisionProcess
from markov_decision_processes.occupancy_measure import add_occupancy_polytope_constraints


from gurobipy import GRB
import gurobipy as gp
import numpy as np


def compute_optimal_policy_occupancy(mdp, reward, log_verbose=False):
    assert reward.shape == (mdp.n_states, mdp.n_actions)
    model = init_gurobi_model(name=f'agent_optimal')
    d_pi_vars = add_occupancy_polytope_constraints(model, mdp)
    objective = gp.quicksum(
        reward[s, a] * d_pi_vars[s, a] / config.SCALING_D_PI
        for a in range(mdp.n_actions)
        for s in range(mdp.n_states)
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Computing max reward failed. Status: {model.status}.")
    max_expected_reward = model.getObjective().getValue()
    optimal_occupancy_measure = np.array([
        [d_pi_vars[s, a].x / config.SCALING_D_PI for a in range(mdp.n_actions)] for s in range(mdp.n_states)
    ])
    if log_verbose:
        print(f"Computed the optimal policy with reward: {max_expected_reward}")
        print(f"\tOpt oc. m.: {optimal_occupancy_measure}")
        print(f"\tOpt policy: {occupancy_to_policy(optimal_occupancy_measure)}")
    return max_expected_reward, optimal_occupancy_measure


def occupancy_to_policy(occupancy_vec):
    probs = occupancy_vec.copy()
    probs[probs.sum(axis=1) == 0, :] = 1
    optimal_policy = np.divide(probs, probs.sum(axis=1)[:, np.newaxis])
    return optimal_policy


if __name__ == '__main__':
    mdp = MarkovDecisionProcess(
        states=[0, 1],
        actions=[0, 1],
        transition_matrix=[
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]]
        ]
    )
    reward_vec = np.array([[0.6, 0.8], [0.5, 0.5]])
    print(compute_optimal_policy_occupancy(mdp, reward_vec, log_verbose=True))
