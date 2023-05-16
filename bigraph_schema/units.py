from pint import UnitRegistry

units = UnitRegistry()


def render_units_type(u):
    dimensionality = u.dimensionality
    unit_keys = list(dimensionality.keys())
    unit_keys.sort()

    numerator = []
    denominator = []

    for unit_key in unit_keys:
        inner_key = unit_key.strip('[]')
        power = dimensionality[unit_key]
        if power > 0:
            if power > 1:
                render = f'{inner_key}**{power}'
            else:
                render = inner_key
            numerator.append(render)
        else:
            power = -power
            if power > 1:
                render = f'{inner_key}**{power}'
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


def test_units_render():
    render = render_units_type(units.newton)
    assert render == 'length*mass/time**2'


if __name__ == '__main__':
    test_units_render()



