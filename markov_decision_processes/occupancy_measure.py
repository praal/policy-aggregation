import gurobipy as gp
from gurobipy import GRB

from config import config


def add_occupancy_polytope_constraints(
        model, mdp, variable_prefix='',
        discounted_setting=None):
    if discounted_setting is not None:
        raise NotImplementedError()
    else:
        # This is the average-reward case
        pass

    d_pi = model.addVars(
        mdp.n_states, mdp.n_actions,
        lb=0, ub=config.SCALING_D_PI, vtype=GRB.CONTINUOUS,
        name="d_pi"
    )
    model.addConstr(
        gp.quicksum(d_pi) == config.SCALING_D_PI, name="occupancy_is_distribution"
    )
    for s in range(mdp.n_states):
        out_flow_s = gp.quicksum(d_pi[s, a] for a in range(mdp.n_actions))
        in_flow_s = gp.quicksum(
            mdp.tr[s_prime, a, s] * d_pi[s_prime, a]
            for a in range(mdp.n_actions)
            for s_prime in range(mdp.n_states)
        )
        model.addConstr(
            out_flow_s == in_flow_s,
            name=f"state_{s}_in_flow_out_flow_constraints"
        )
    return d_pi

