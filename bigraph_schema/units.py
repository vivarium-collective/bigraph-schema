"""
=====
Units
=====

Register all of the unit types from the pint unit registry
"""

from pint import UnitRegistry


units = UnitRegistry()

# Module-level handle that ``Quantity``'s realize/serialize use to
# build ``pint.Quantity`` instances. Defaults to bigraph-schema's own
# registry; downstream packages can swap it out via
# ``set_quantity_registry`` so quantities they construct elsewhere
# (e.g. ``vivarium.library.units``) interoperate with framework code.
_quantity_registry = units


def get_quantity_registry():
    return _quantity_registry


def set_quantity_registry(registry):
    """Override the registry used by ``Quantity`` realize/serialize.

    Pint's ``Quantity`` instances are tied to the registry that created
    them; mixing instances from different registries raises. Set this
    to your project's registry at import time."""
    global _quantity_registry
    _quantity_registry = registry


def parse_time_interval(value):
    """Normalize a declared timestep to (canonical_seconds, declared_unit).

    ``value`` may be a number (interpreted as seconds, declared_unit None) or a
    pint-parseable time string ("1 min", "2 minutes", "30 s"). A non-time
    quantity or an unparseable string raises ValueError naming the value.
    """
    if isinstance(value, bool):
        raise ValueError(f'interval must be a number or time string, got {value!r}')
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        ureg = get_quantity_registry()
        try:
            quantity = ureg.Quantity(value)
        except Exception as error:  # pint raises several types; normalize them
            raise ValueError(
                f'could not parse interval {value!r} as a time quantity: {error}')
        if not quantity.check('[time]'):
            raise ValueError(
                f'interval {value!r} is not a time quantity '
                f'(dimensionality {quantity.dimensionality})')
        return float(quantity.to('second').magnitude), str(quantity.units)
    raise ValueError(f'interval must be a number or time string, got {value!r}')


def format_time_interval(seconds, unit):
    """Render the author-facing label for a canonical-seconds interval.

    ``unit is None`` -> "<seconds> s" (canonical). Otherwise convert back into
    the declared ``unit`` and append its short symbol, e.g. "1 min".
    """
    if unit is None:
        return f'{seconds:g} s'
    ureg = get_quantity_registry()
    magnitude = ureg.Quantity(seconds, 'second').to(unit).magnitude
    symbol = f'{ureg.Unit(unit):~}'
    return f'{magnitude:g} {symbol}'


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


def test_parse_time_interval():
    # bare number -> seconds, no declared unit
    assert parse_time_interval(60) == (60.0, None)
    assert parse_time_interval(1.5) == (1.5, None)

    # pint strings -> canonical seconds + declared unit name
    assert parse_time_interval('1 min') == (60.0, 'minute')
    assert parse_time_interval('2 minutes') == (120.0, 'minute')
    assert parse_time_interval('30 s') == (30.0, 'second')
    assert parse_time_interval('1.5 min') == (90.0, 'minute')

    # non-time dimensionality is rejected, naming the value
    import pytest
    with pytest.raises(ValueError, match='1 meter'):
        parse_time_interval('1 meter')
    with pytest.raises(ValueError, match='nonsense'):
        parse_time_interval('nonsense')


def test_format_time_interval():
    assert format_time_interval(60.0, 'minute') == '1 min'
    assert format_time_interval(120.0, 'minute') == '2 min'
    assert format_time_interval(90.0, 'minute') == '1.5 min'
    assert format_time_interval(30.0, 'second') == '30 s'
    # no declared unit -> canonical seconds label
    assert format_time_interval(90.0, None) == '90 s'
    assert format_time_interval(0.5, None) == '0.5 s'


def test_time_interval_round_trip():
    # parse then format returns the author's declared form
    seconds, unit = parse_time_interval('1 min')
    assert format_time_interval(seconds, unit) == '1 min'
    seconds, unit = parse_time_interval(60)
    assert format_time_interval(seconds, unit) == '60 s'


if __name__ == '__main__':
    test_units_render()
    test_roots_cycle()
