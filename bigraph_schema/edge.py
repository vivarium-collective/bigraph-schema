"""
====
Edge
====

Base class for all edges in the bigraph schema.
"""

def default_wires(schema):
    """
    Create default wiring for a schema by connecting each port to a store of the same name.
    """
    return {
        key: [key]
        for key in schema}


class Edge:
    """
    Base class for all computational edges in the bigraph schema.

    Edges define the interface between simulation processes and the global state,
    specifying the structure of input and output ports using bigraph types
    (e.g., 'float', 'map[float]', 'list[integer]', etc.).

    Upon instantiation, each edge registers its port types with the core.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        """
        Initialize the edge with a config and simulation core.

        Args:
            config (dict): Optional configuration dictionary. Defaults to empty dict.
            core: The core simulation engine that manages this edge and its types.

        Raises:
            Exception: If `core` is not provided.
        """
        if core is None:
            raise Exception('must provide a core')
        self.core = core

        if config is None:
            config = {}

        self._config = self.core.fill(self.config_schema, config)
        self._composition = 'edge'
        self._state = {}

        self.initialize(
            self._config)


    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, composition):
        self._composition = composition

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def initialize(self, config):
        """Optional hook for subclass-specific initialization."""
        pass

    def initial_state(self):
        """Return initial state values, if applicable."""
        return {}

    @staticmethod
    def generate_state(config=None):
        """Generate static initial state for user configuration or inspection."""
        return {}

    def inputs(self):
        """
        Return a dictionary mapping input port names to bigraph types.

        Example:
            {'glucose': 'float', 'biomass': 'map[float]'}
        """
        return {}

    def outputs(self):
        """
        Return a dictionary mapping output port names to bigraph types.

        Example:
            {'growth_rate': 'float'}
        """
        return {}

    def default_config(self):
        """ get the default of the config_schema for this edge """
        return self.core.default(
            self.config_schema)

    def default_inputs(self):
        """Generate default wire paths for inputs: {port: [port]}"""
        return default_wires(self.inputs())

    def default_outputs(self):
        """Generate default wire paths for outputs: {port: [port]}"""
        return default_wires(self.outputs())

    def interface(self):
        """
        Return combined interface schema as a dict:
            {
                'inputs': {port: type, ...},
                'outputs': {port: type, ...}
            }
        """
        return {
            'inputs': self.inputs(),
            'outputs': self.outputs()
        }


