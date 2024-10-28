import os
import pickle

from config import config

EXP_RESULTS_PATH = os.path.join(config.ROOT_DIR, 'exp_results')


def _save_file_path(name):
    f_name = f"{name}.pkl"
    return os.path.join(EXP_RESULTS_PATH, f_name)


def load_instance_by_name(name):
    with open(_save_file_path(name), "rb") as f:
        return pickle.load(f)


class ExperimentObject(object):
    def __init__(self, name):
        self.name = name

    def save_snapshot(self):
        with open(_save_file_path(self.name), "wb") as f:
            pickle.dump(self, f)


def load_env_from_name(env_name):
    env = load_instance_by_name(env_name)
    return env


def load_env_from_path(f_path):
    env_name = env_name_from_path(f_path)
    return load_env_from_name(env_name)


def env_name_from_path(f_path):
    return f_path.split('/')[-1].split('.pkl')[0]
