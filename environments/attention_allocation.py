import numpy as np

from experiments.base_experiment import ExperimentObject
from markov_decision_processes.markov_decision_process import MarkovDecisionProcess
from datetime import datetime
from itertools import product


class AttentionAllocation(ExperimentObject):
    def __init__(self, n_factories, n_phases, transition_phases=None):
        _date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        _rand = np.random.randint(int(1e5), int(1e6))
        name = f'factory_{n_factories}_{n_phases}_object_{_date}_{_rand}'
        super().__init__(name)
        self.n_factories = n_factories
        self.n_phases = n_phases

        cartesian_product = product(list(range(n_phases)), repeat=n_factories)
        self.states = dict([(x[1], x[0]) for x in enumerate(cartesian_product)])

        self.actions = list(range(self.n_factories + 1))
        self.phase_trans = list(
            list(np.random.randint(50, 80) / 100 for _ in range(self.n_phases))
            for _ in range(self.n_factories)
        )

        self.tr = None
        self.create_transition_function()

        self.mdp = MarkovDecisionProcess(
            list(self.states.values()),
            list(range(self.n_actions)),
            self.tr
        )

    def create_transition_function(self):
        self.tr = np.zeros((self.n_states, self.n_actions, self.n_states))
        for state, s_id in self.states.items():
            for a_id, a in enumerate(self.actions):
                next_state_probs = self.find_next_states(state, a)
                for s_end, prob in next_state_probs:
                    self.tr[s_id, a_id, self.states[s_end]] = prob


    def sample_policies(self, n_samples):
        self.mdp.sample_policies_randomly(n_samples)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_actions(self):
        return len(self.actions)

    def find_next_states(self, s, a):
        s_next_fixed = [None for _ in range(len(s))]
        fixed_fs = list()
        for f in range(self.n_factories):
            if s[f] == self.n_phases - 1:
                s_next_fixed[f] = 0 if a == f else s[f]
                fixed_fs.append(f)
            if f == a:
                s_next_fixed[f] = max(0, s[f] - 1)
                fixed_fs.append(f)
        all_next_states = list()
        for next_s in self.states.keys():
            if sum(next_s[x] == s_next_fixed[x] for x in fixed_fs) != len(fixed_fs):
                continue
            trans_prob = 1
            for i in range(self.n_factories):
                if i in fixed_fs:
                    continue
                if next_s[i] < s[i] or next_s[i] > s[i] + 1:
                    trans_prob = None
                    break
                elif next_s[i] == s[i] + 1:
                    trans_prob *= self.phase_trans[i][s[i]]
                elif next_s[i] == s[i]:
                    trans_prob *= 1 - self.phase_trans[i][s[i]]
                else:
                    raise RuntimeError("Should not reach here")
            if trans_prob is not None:
                all_next_states.append((next_s, trans_prob))
        total_prob = sum(x[1] for x in all_next_states)
        for i, x in enumerate(all_next_states):
            all_next_states[i] = x[0], x[1] / total_prob
        return all_next_states


if __name__ == '__main__':
    AttentionAllocation(5, 3).mdp.sample_policies_randomly(10000)
