"""
====
Edge
====

Base class for all edges in the bigraph schema.
"""

class Edge:
    def __init__(self):
        pass


    def inputs(self):
        return {}


    def outputs(self):
        return {}


    def interface(self):
        """Returns the schema for this type"""
        return {
            'inputs': self.inputs(),
            'outputs': self.outputs()}
