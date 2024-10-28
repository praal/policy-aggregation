import gurobipy as gp


def init_gurobi_model(name='policy_aggr_lp', gurobi_log=0):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", gurobi_log)
    env.start()
    model = gp.Model(name, env=env)
    return model
