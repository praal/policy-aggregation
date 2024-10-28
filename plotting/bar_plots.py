from experiments.base_experiment import load_env_from_path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem


rules_to_plot = [
    'quantile',
    'borda',
    # '0.9500000000000001-approvals',
    '0.9-approvals',
    '0.8-approvals',
    'egalitarian',
    '0.0-approvals',
]

final_rule_name = {
    'quantile': 'max-q',
    'borda': 'borda',
    'egalitarian': 'egal',
    '0.0-approvals': 'util',
    '0.8-approvals': '80%-a',
    '0.9-approvals': '90%-a',
    # '0.9500000000000001-approvals': '95%-a'
}


def plot_bars_utilities(directory, exp_prefix):
    testing_add_dummy_noise = False

    import os

    matching_files = [f for f in os.listdir(directory) if f.startswith(exp_prefix)]
    agg_vals = dict()
    for exp_path in matching_files:
        print("adding", exp_path)
        exp_instance = load_env_from_path(exp_path)
        for r_name, res in exp_instance.exp_results.items():
            vals = []
            for ag in exp_instance.agents:
                v = ag.evaluate_policy_occupancy(res['occupancy'])
                vals.append((v - ag.min_expected_reward) / (
                        ag.max_expected_reward - ag.min_expected_reward))
            vals = list(sorted(vals))
            if r_name not in agg_vals:
                agg_vals[r_name] = list()
            agg_vals[r_name].append([
                x + ((np.random.rand() - 0.5) / 10 if testing_add_dummy_noise else 0) for x in vals])
        del exp_instance
    print(*agg_vals.keys(), sep='\n')

    all_colors_plt = list(sns.color_palette(
        # 'tab10',
        'deep',
        as_cmap=False).as_hex())
    plt.rcParams.update({'font.size': 12})
    plt.margins(x=0.025, y=0.05)
    plt.rcParams['axes.xmargin'] = 0.025
    plt.rcParams['axes.ymargin'] = 0.05

    num_rules = len(rules_to_plot)

    m = len(list(agg_vals.values())[0][0])
    x = np.arange(1, m + 1)
    if True:
        bar_width = 0.8 / num_rules
        fig, ax = plt.subplots(figsize=(5, 2.5))

        for i, rule_name in enumerate(rules_to_plot):
            values = agg_vals[rule_name]
            # Calculate position for each bar group
            bar_positions = x - 0.4 + i * bar_width
            to_plot_vals = [np.mean([100 * x[j] for x in values]) for j in range(m)]
            if len(list(agg_vals.values())[0]) == 1:
                err_vals = 0
            else:
                err_vals = [sem([100 * x[j] for x in values]) for j in range(m)]
            # print(to_plot_vals, err_vals)
            # exit(1)
            ax.bar(bar_positions, to_plot_vals, bar_width * 3 / 4,
                   label=final_rule_name[rule_name],
                   yerr=err_vals,
                   color=all_colors_plt[i])

        ax.set_xlabel('Ranked by % Max Return')
        ax.set_ylabel('% Max Return')
        ax.set_xticks(x)
        ax.legend(ncol=2, framealpha=0.2, handletextpad=0.2, labelspacing=0.2)

        # Show plot
        plt.savefig('one-sample-base-factories.pdf', bbox_inches="tight")
    if True:
        fig, ax = plt.subplots(figsize=(5, 2.5))

        x = np.arange(num_rules)
        bar_width = 0.8 / m  # Adjust this to fit all bars

        for i in range(m):
            bar_positions = x + (i * bar_width)
            # Calculate position for each bar group
            to_plot_vals = [np.mean([100 * _vals[i] for _vals in agg_vals[rule]]) for rule in rules_to_plot]
            if len(list(agg_vals.values())[0]) == 1:
                err_vals = 0
            else:
                err_vals = [sem([100 * _vals[i] for _vals in agg_vals[rule]]) for rule in rules_to_plot]
            ax.bar(bar_positions, to_plot_vals, bar_width * 3 / 4,
                   yerr=err_vals,
                   color=[all_colors_plt[i] for i in range(len(rules_to_plot))])

        ax.set_ylabel('% Max Return')
        ax.set_xticks(x + bar_width * (m - 1) / 2)
        ax.set_xticklabels([final_rule_name[r] for r in rules_to_plot])

        # Show plot
        plt.savefig('one-sample-base-factories-per-rule.pdf', bbox_inches="tight")


