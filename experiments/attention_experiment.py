from tqdm import tqdm

from environments.attention_allocation import AttentionAllocation
import numpy as np

from experiments.base_experiment import ExperimentObject, load_env_from_name
from markov_decision_processes.agent import Agent
from mechanisms.approval_rules import ApprovalsRule
from mechanisms.borda import BordaRule
from mechanisms.egalitarian import EgalitarianRule
from mechanisms.max_quantile import MaxQuantileRule


class AttentionExperiment(ExperimentObject):
    def __init__(self, env_name):
        exp_name = f'exp_{env_name}_results'
        super().__init__(exp_name)
        self.attention_env = load_env_from_name(env_name)
        self.agents = list()
        self.base_agents = list()
        self.fire = -100
        self.exp_results = dict()

    def reset_experiment(self):
        self.agents = list()
        self.base_agents = list()

    def add_agent(self, targets, agent_weight):
        state_ids = dict()
        for key, val in self.attention_env.states.items():
            state_ids[val] = key
        new_reward = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                if a < self.attention_env.n_factories:
                    new_reward[s][a] = -1
                for t in targets:
                    if state_ids[s][t] == self.attention_env.n_phases - 1 and a != t:
                        new_reward[s][a] += self.fire * agent_weight[t]
        # print("agent targets", targets, "----", new_reward)
        t_agent = Agent(new_reward, f'{len(self.agents)}_' + '-'.join(str(x) for x in targets))
        self.agents.append(t_agent)

    def add_base_agents(self, scaling_noise=False):
        state_ids = dict()
        for key, val in self.attention_env.states.items():
            state_ids[val] = key
        for i in range(self.attention_env.n_factories):
            noise_i = np.random.randint(1, 4 * (self.attention_env.n_factories + 1)) / 4 if scaling_noise else 1
            # noise_i = (i + 1)
            print("noise", i, noise_i)
            new_reward = np.zeros((self.mdp.n_states, self.mdp.n_actions))
            for s in range(self.mdp.n_states):
                for a in range(self.mdp.n_actions):
                    if a < self.attention_env.n_factories:
                        new_reward[s][a] = -1
                    if state_ids[s][i] == self.attention_env.n_phases - 1 and a != i:
                        new_reward[s][a] += self.fire * noise_i

            # print("agent", i, "----", new_reward)
            t_agent = Agent(new_reward, "base" + str(i))
            self.base_agents.append(t_agent)

    @property
    def mdp(self):
        return self.attention_env.mdp

    def setup_agent_utilities(self):
        for a in self.agents:
            a.compute_opt_policy(self.mdp, fact_env=self.attention_env)
        print("evaluating samples for agents")
        for i, (p, p_state_occ) in tqdm(enumerate(zip(
                self.mdp.sampled_policies,
                self.mdp.sampled_state_occupancies
        ))):
            state_action_occupancy = p * p_state_occ[:, :, np.newaxis]
            for a in self.agents:
                a.add_sampled_policy_returns(state_action_occupancy)
        for a in self.agents:
            # for a2 in self.agents:
            for a2 in [a]:
                a2.add_sampled_policy_returns(a.opt_policy_occ[np.newaxis, :])
                a2.add_sampled_policy_returns(a.worst_policy_occ[np.newaxis, :])
        for a in self.agents:
            a.flatten_sort_sampled_returns()

    def run_rules(self):
        rules = [
            EgalitarianRule,
            BordaRule,
            MaxQuantileRule,
        ]
        # alphas_approvals = list(i / 10 for i in range(11))
        # alphas_approvals += list(i / 100 + 0.81 for i in range(10))
        # alphas_approvals += list(i / 100 + 0.91 for i in range(10))
        alphas_approvals = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1]
        print(alphas_approvals)
        rules_initialized = list()
        for alpha in alphas_approvals:
            rules_initialized.append(ApprovalsRule(self.mdp, self.agents, alpha))
        for r in rules:
            rules_initialized.append(r(self.mdp, self.agents))
        for r in rules_initialized:
            info, occupancy_measure = r.solve()
            self.exp_results[r.name] = {
                'info': info,
                'occupancy': occupancy_measure
            }
            print("Rule", r.name, "info", info)
            for a in self.agents:
                print(f"\t\tAgent {a.id}, "
                      f"expected return {a.evaluate_policy_occupancy(occupancy_measure)}")


def base_agent_experiment(exp_name):
    exp_instance = AttentionExperiment(exp_name)
    exp_instance.add_base_agents(scaling_noise=True)
    exp_instance.agents = exp_instance.base_agents.copy()
    exp_instance.setup_agent_utilities()
    exp_instance.run_rules()
    exp_instance.save_snapshot()


def intersecting_agent_experiment(exp_name):
    def get_set_bit_indices(n):
        indices = []
        _i = 0
        while n:
            if n & 1:
                indices.append(_i)
            n >>= 1
            _i += 1
        return indices

    exp_instance = AttentionExperiment(exp_name)
    exp_instance.name = f'subsets_{exp_instance.name}'
    exp_instance.add_base_agents(scaling_noise=True)
    n_factories = exp_instance.attention_env.n_factories
    weights = [np.random.randint(2, n_factories + 1) / 2 for _ in range(n_factories)]
    print(weights)
    for i in range(10):
        noise_i = np.random.randint(1, 4 * (n_factories + 1)) / 4
        targets = get_set_bit_indices(np.random.randint(1, 2 ** n_factories))
        agent_weights = [weights[f] * noise_i for f in range(n_factories)]
        print(f"agent {i}, target: {targets}, weights: {agent_weights}")
        exp_instance.add_agent(targets, agent_weights)
    exp_instance.setup_agent_utilities()
    exp_instance.run_rules()
    exp_instance.save_snapshot()


