import sys

from environments.attention_allocation import AttentionAllocation
from experiments.attention_experiment import base_agent_experiment, intersecting_agent_experiment
from experiments.base_experiment import load_env_from_path, env_name_from_path
from plotting.bar_plots import plot_bars_utilities


def create_fact_mdp():
    n_fact, n_phase = int(sys.argv[2]), int(sys.argv[3])
    env = AttentionAllocation(n_fact, n_phase)
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    env.sample_policies(n_samples)
    env.save_snapshot()


def add_samples():
    env = load_env_from_path(sys.argv[2])
    n_samples = int(sys.argv[3])
    env.sample_policies(n_samples)
    env.save_snapshot()


def report():
    env = load_env_from_path(sys.argv[2])
    print(f"Factory Attention Allocation, #factories: {env.n_factories} #phases: {env.n_phases}")
    print("\t #samples:", sum(x.shape[0] for x in env.mdp.sampled_state_occupancies))


def run_base_experiments():
    exp_name = env_name_from_path(sys.argv[2])
    base_agent_experiment(exp_name)


def run_subsets_experiments():
    exp_name = env_name_from_path(sys.argv[2])
    intersecting_agent_experiment(exp_name)


def create_bar_plots():
    if len(sys.argv) < 4:
        raise NotImplementedError("At least 4 inputs, with folder and prefix for plotting")
    plot_bars_utilities(sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise NotImplementedError("At least 2 inputs")

    if sys.argv[1] == 'create':
        create_fact_mdp()
    elif sys.argv[1] == 'samples':
        # raise NotImplementedError()
        add_samples()
    elif sys.argv[1] == 'report':
        # raise NotImplementedError
        report()
    elif sys.argv[1] == 'base':
        run_base_experiments()
    elif sys.argv[1] == 'subsets':
        run_subsets_experiments()
    elif sys.argv[1] == 'plot':
        create_bar_plots()
    else:
        raise NotImplementedError("command not implemented")
