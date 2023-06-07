"""
=====
Units
=====

Register all of the unit types from the pint unit registry
"""

from pint import UnitRegistry


units = UnitRegistry()


def render_coefficient(original_power):
    power = abs(original_power)
    int_part = int(power)
    root_part = power % 1

    if root_part != 0.0:
        render = str(root_part)[2:]
        render = f'{int_part}_{render}'
    else:
        render = str(int_part)

    return render


def render_units_type(dimensionality):
    unit_keys = list(dimensionality.keys())
    unit_keys.sort()

    numerator = []
    denominator = []

    for unit_key in unit_keys:
        inner_key = unit_key.strip('[]')
        power = dimensionality[unit_key]
        negative = False
        
        if power < 0:
            negative = True
            power = -power

        if power == 1:
            render = inner_key
        else:
            render = f'{inner_key}^{render_coefficient(power)}'

        if negative:
            denominator.append(render)
        else:
            numerator.append(render)

    render = '*'.join(numerator)
    if len(denominator) > 0:
        render_denominator = '*'.join(denominator)
        render = f'{render}/{render_denominator}'

    return render


def parse_coefficient(s):
    if s is None:
        return 1
    elif '_' in s:
        parts = s.split('_')
        if len(parts) > 1:
            base, residue = parts
            return int(base) + (float(residue) / 10.0)
        else:
            return int(parts)
    else:
        return int(s)


def parse_dimensionality(s):
    numerator, denominator = s.split('/')
    numerator_terms = numerator.split('*')
    denominator_terms = denominator.split('*')

    dimensionality = {}

    for term in numerator_terms:
        base = term.split('^')
        exponent = None
        if len(base) > 1:
            exponent = base[1]
        dimensionality[f'[{base[0]}]'] = parse_coefficient(exponent)

    for term in denominator_terms:
        power = term.split('^')
        exponent = None
        if len(power) > 1:
            exponent = power[1]
        dimensionality[f'[{power[0]}]'] = -parse_coefficient(exponent)

    return dimensionality


def test_units_render():
    dimensionality = units.newton.dimensionality
    render = render_units_type(dimensionality)
    recover = parse_dimensionality(render)

    print(f'original: {dimensionality}')
    print(f'render: {render}')
    print(f'parsed: {recover}')

    assert render == 'length*mass/time^2'
    assert recover == dimensionality


def test_roots_cycle():
    dimensionality = {
        '[length]': 1.5,
        '[time]': 3,
        '[mass]': -2.5,
    }
    render = render_units_type(dimensionality)
    recover = parse_dimensionality(render)

    print(f'original: {dimensionality}')
    print(f'render: {render}')
    print(f'parsed: {recover}')

    assert render == 'length^1_5*time^3/mass^2_5'
    assert recover == dimensionality


if __name__ == '__main__':
    test_units_render()
    test_roots_cycle()
