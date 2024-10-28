import numpy as np
import gurobipy as gp
from tqdm import tqdm

import concurrent.futures
import threading


class MarkovDecisionProcess(object):
    EPSILON = 1e-5

    def __init__(self, states, actions, transition_matrix):
        self.states = states
        self.actions = actions
        self.tr = np.array(transition_matrix)

        # assuming the initial distribution is uniform over all states
        self.s0 = np.ones(self.n_states, dtype=np.float64)
        self.s0 /= self.s0.sum()
        self.rewards = dict()

        self.id = f'mdp_{np.random.randint(int(1e8), int(1e9))}'

        self.sampled_policies = list()
        self.sampled_state_occupancies = list()

        self.occupancy_lp = None

        assert (self.n_states, self.n_actions, self.n_states) == self.tr.shape

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_states(self):
        return len(self.states)

    def sample_policies_randomly(self, n_samples, n_workers=6):
        union_samples = list()
        worker_samples = n_samples // n_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._sample_policies_randomly, worker_samples) for _ in range(n_workers)]
            for future in concurrent.futures.as_completed(futures):
                union_samples.append(future.result())
        for sampled_ps, sampled_state_occupancies in union_samples:
            self.sampled_policies.extend(sampled_ps)
            self.sampled_state_occupancies.extend(sampled_state_occupancies)

    def _sample_policies_randomly(self, n_samples):
        batch_size = 50
        all_samples_state_occupancies = list()
        all_samples_policies = list()
        for _ in tqdm(range(n_samples // batch_size)):
            new_sample_policies = np.random.dirichlet(
                np.ones(self.n_actions),
                size=(batch_size, self.n_states))
            all_samples_policies.append(new_sample_policies)
            state_trans = (new_sample_policies[:, :, :, np.newaxis] * self.tr[np.newaxis, :, :, :]).sum(axis=2)
            mean_state_trans = 0.5 * (
                np.tile(
                    np.eye(self.n_states)[np.newaxis, :, :],
                    (batch_size, 1, 1)
                ) + state_trans
            )
            final_mean_state_trans = state_trans
            early_convergence = False
            for i in range(30):
                state_trans = np.matmul(state_trans, state_trans)
                new_final_mean_state_trans = np.matmul(state_trans, mean_state_trans)
                mean_state_trans = 0.5 * mean_state_trans + 0.5 * new_final_mean_state_trans
                delta = np.max(np.abs(new_final_mean_state_trans - final_mean_state_trans))
                final_mean_state_trans = new_final_mean_state_trans
                if delta < 1e-8:
                    early_convergence = True
                    break
            if not early_convergence and delta > 1e-4:
                raise RuntimeError(f"Stopped too early, delta = {delta}")
            state_occupancy = (final_mean_state_trans * self.s0[np.newaxis, :]).sum(axis=1)

            """ Important: state_action_occupancy = new_sample_policies * state_occupancy[:, :, np.newaxis] """

            all_samples_state_occupancies.append(state_occupancy)
        return all_samples_policies, all_samples_state_occupancies

    @staticmethod
    def get_state_action_occupancy(state_occupancies, policy):
        return policy * state_occupancies[:, :, np.newaxis]
