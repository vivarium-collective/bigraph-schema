

def react_divide(config):
    return {
        'redex': {
            config['id']: {}},
        'reactum': {
            daughter_config['id']: daughter_config['state']
            for daughter_config in config['daughters']}}


