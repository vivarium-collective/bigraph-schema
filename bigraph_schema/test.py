from bigraph_schema import TypeSystem


def schema_zoo():
    mitochondria_schema = {
        'mitochondria': {
            'volume': {'_type': 'float'},
            'membrane': {
                'surface_proteins': {'_type': 'tree[protein]'},
                'potential': {'_type': 'microvolts'}},
            'mass': {'_type': 'membrane?'},
        }
    }

    cytoplasm_schema = {
        'cytoplasm': {
            'mitochondria': {'_type': 'tree[mitochondria]'},
            'proteins': {'_type': 'tree[mitochondria]'},
            'nucleus': {'_type': 'tree[mitochondria]'},
            'transcripts': {'_type': 'tree[mitochondria]'},
        }
    }

    cell_schema = {
        'cell': {
            'shape': {'_type': 'mesh'},
            'volume': {'_type': 'mL'},
            'temperature': {'_type': 'K'},
        }
    }

    cell_composite = {
        'environment': {
            'outer_shape': {
                '_type': 'mesh', '_value': []},
            'cellA': {
                'cytoplasm': {
                    'external_ions': {'_type': 'ions'},
                    'internal_ions': {'_type': 'ions'},
                    'other_ions': {'_type': {
                        '_default': 0.0,
                        '_apply': accumulate,
                        '_serialize': str,
                        '_deserialize': float,
                        '_divide': divide_float,
                        '_description': '64-bit floating point precision number'
                    }},
                    'electron_transport': {
                        '_type': 'process',
                        '_value': 'ElectronTransport',
                        '_ports': {
                            'external_ions': 'ions',
                            'internal_ions': 'ions'},
                        '_wires': {
                            'external_ions': ['..', 'external_ions'],
                            'internal_ions': ['..', 'internal_ions']}
                    }
                },
                'inner_shape': {'_type': 'mesh', '_value': []},
                '_ports': {
                    'shape': 'mesh',
                    'volume': 'mL',
                    'temperature': 'K'
                },
                '_channel': {
                    'shape': ['inner_shape'],
                },
                '_wires': {
                    'shape': ['..', 'outer_shape']
                }
            }
        }
    }

    compose({
        'cell': {
            'membrane': cell_schema,
            'cytoplasm': cytoplasm_schema
        }
    }, {

    })


def run_register(base_types):
    # print(types)
    x=1
    return {}


if __name__ == '__main__':
    base_types = TypeSystem()

    run_register(base_types)
