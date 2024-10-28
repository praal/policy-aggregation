import numpy as np

from markov_decision_processes.agent_optimal_policy import compute_optimal_policy_occupancy, occupancy_to_policy
from markov_decision_processes.markov_decision_process import MarkovDecisionProcess
from scipy.stats import genlogistic


class Agent(object):
    def __init__(self, reward, agent_id=None):
        self.reward = np.array(reward)

        if agent_id is None:
            agent_id = np.random.randint(int(1e8), int(1e9))
        self.id = f'agent_{agent_id}'
        self.mdp_id = None

        self.max_expected_reward = None
        self.min_expected_reward = None
        self.opt_policy_occ = None
        self.worst_policy_occ = None
        self.sampled_policy_returns = list()
        self.histogram_policy_returns = None

        self.all_returns_not_sorted = None
        self.gen_log_params = None
        self.gen_log_mode = None

    def compute_opt_policy(self, mdp, **kwargs):
        if self.mdp_id is not None and mdp.id != self.mdp_id:
            raise ValueError("Already set to a different mdp")

        self.mdp_id = mdp.id

        min_reward, worst_occupancy = compute_optimal_policy_occupancy(mdp, -self.reward)
        min_reward *= -1.0
        self.worst_policy_occ = worst_occupancy
        max_reward, optimal_occupancy = compute_optimal_policy_occupancy(mdp, self.reward)
        self.opt_policy_occ = optimal_occupancy
        if max_reward < min_reward + 1e-4:
            raise RuntimeError("Min and max average reward are too close.")
        self.min_expected_reward = min_reward
        self.max_expected_reward = max_reward
        print("agent", self.id)

    def add_sampled_policy_returns(self, state_action_occupancy):
        p_returns = (state_action_occupancy * self.reward).sum(axis=(1, 2))
        self.sampled_policy_returns.append(p_returns)

    def evaluate_policy_occupancy(self, state_action_occupancy):
        return (state_action_occupancy * self.reward).sum()

    def flatten_sort_sampled_returns(self):
        self.all_returns_not_sorted = np.concatenate(self.sampled_policy_returns)
        self.histogram_policy_returns = np.sort(self.all_returns_not_sorted)

        self.gen_log_params = genlogistic.fit(self.all_returns_not_sorted)
        precision = 0.001
        low, high = self.min_expected_reward, self.max_expected_reward
        while high - low > precision:
            mid = (low + high) / 2
            mid_value = genlogistic.pdf(mid, *self.gen_log_params)
            mid_plus_one_value = genlogistic.pdf(mid + precision, *self.gen_log_params)

            if mid_value > mid_plus_one_value:
                high = mid
            else:
                low = mid
        self.gen_log_mode = high
        print("fitted data", self.gen_log_mode, self.gen_log_params)

    def alpha_quantile_utility(self, alpha):
        if self.histogram_policy_returns is None:
            self.flatten_sort_sampled_returns()
        sample_size = self.histogram_policy_returns.size
        percentile_index = int(sample_size * alpha)
        if percentile_index == sample_size:
            return self.max_expected_reward
        return self.histogram_policy_returns[percentile_index]


if __name__ == '__main__':
    mdp = MarkovDecisionProcess(
        states=[0, 1],
        actions=[0, 1],
        transition_matrix=[
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]]
        ]
    )
    agent = Agent([[0, 1], [0, 1]])
    agent.compute_opt_policy(mdp)
