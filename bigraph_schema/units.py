from pint import UnitRegistry


units = UnitRegistry()


def render_units_type(dimensionality):
    # dimensionality = unit.dimensionality
    unit_keys = list(dimensionality.keys())
    unit_keys.sort()

    numerator = []
    denominator = []

    for unit_key in unit_keys:
        inner_key = unit_key.strip('[]')
        power = dimensionality[unit_key]
        if power % 1 == 0:
            power = int(power)
        if power > 0:
            if power > 1:
                render = f'{inner_key}^{power}'
            else:
                render = inner_key
            numerator.append(render)
        else:
            power = -power
            if power > 1:
                render = f'{inner_key}^{power}'
            else:
                render = inner_key
            denominator.append(render)

    render = '*'.join(numerator)
    if len(denominator) > 0:
        render_denominator = '*'.join(denominator)
        if len(denominator) > 1:
            render_denominator = f'({render_denominator})'
        render = f'{render}/{render_denominator}'

    return render


def parse_dimensionality(s):
    numerator, denominator = s.split('/')
    numerator_terms = numerator.split('*')
    denominator_terms = denominator.split('*')

    dimensionality = {}

    for term in numerator_terms:
        power = term.split('^')
        exponent = 1
        if len(power) > 1:
            exponent = power[1]
        dimensionality[f'[{power[0]}]'] = int(exponent)

    for term in denominator_terms:
        power = term.split('^')
        exponent = 1
        if len(power) > 1:
            exponent = power[1]
        dimensionality[f'[{power[0]}]'] = -int(exponent)

    return dimensionality


def test_units_render():
    dimensionality = units.newton.dimensionality
    render = render_units_type(dimensionality)
    assert render == 'length*mass/time^2'

    recover = parse_dimensionality(render)
    assert recover == dimensionality


if __name__ == '__main__':
    test_units_render()



