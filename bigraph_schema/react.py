"""
=====
React
=====
"""


def react_divide_counts(config):
    """
    Divides counts among daughters based on their ratios in a reaction configuration.

    This function constructs a reaction configuration to divide counts among daughter entities based on specified
    ratios. If no ratio is specified for a daughter, the counts are divided evenly. The configuration includes the
    setup for input ('redex'), output ('reactum'), and the division operation ('calls').

    Args:
    - config (dict): Configuration dict with keys 'id' (string), 'state_key' (string), 'daughters' (list of dicts).
        Each daughter dict may include 'id' (string) and optionally 'ratio' (float).

    Returns:
    - dict: A dictionary with keys 'redex', 'reactum', and 'calls', detailing the reaction setup.
    """

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
