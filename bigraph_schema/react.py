

def react_divide_counts(config):
    redex = {
        config['id']: {
            config['state_key']: '@counts'}}

    bindings = [
        f'{daughter["id"]}_counts'
        for daughter in config['daughters']]

    reactum = {
        daughter['id']: {
            config['state_key']: binding}
        for binding, daughter in zip(bindings, config['daughters'])}

    even = 1.0 / len(config['daughters'])
    ratios = [
        daughter.get('ratio', even)
        for daughter in config['daughters']]

    calls = [{
        'function': 'divide_counts',
        'arguments': ['@counts', ratios],
        'bindings': bindings}]

    return {
        'redex': redex,
        'reactum': reactum,
        'calls': calls}


