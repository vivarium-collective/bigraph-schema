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

        self.config = self.core.fill(
            self.config_schema,
            config)

        self.initialize(self.config)


    def initialize(self, config):
        pass


    def initial_state(self):
        """The initial state of the edge, which is passed to the core."""
        return {}


    @staticmethod
    def generate_state(config=None):  # TODO -- config could have a schema
        """
        Generate an initial state statically, without any instance-specific config.
        This could be used to create user-configured initial states based on the Edge's requirements.
        """
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
