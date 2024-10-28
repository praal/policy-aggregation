
class AggregationRule(object):
    def __init__(self, mdp, agents):
        self.mdp = mdp
        self.agents = agents
        self.name = None

    def solve(self):
        raise NotImplementedError()

    @property
    def n_agents(self):
        return len(self.agents)
