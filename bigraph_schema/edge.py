"""
====
Edge
====

Base class for all edges in the bigraph schema.
"""

class Edge:
    config_schema = {}


    def __init__(self, config=None, core=None):
        if core is None:
            raise Exception('must provide a core')

        self.core = core

        if config is None:
            config = {}

        _, self.config = self.core.generate(
            self.config_schema,
            config)

        self.initialize(self.config)


    def initialize(self, config):
        pass


    def initial_state(self):
        return {}


    def inputs(self):
        return {}


    def outputs(self):
        return {}


    def interface(self):
        """Returns the schema for this type"""
        return {
            'inputs': self.inputs(),
            'outputs': self.outputs()}
