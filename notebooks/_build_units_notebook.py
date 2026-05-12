"""Build notebooks/units.ipynb from cell sources defined below.

Run from repo root: ``PYTHONPATH=. uv run python3 notebooks/_build_units_notebook.py``
Re-runs are idempotent: overwrites units.ipynb each time.
"""
import json
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Cells: list of (cell_type, source_lines) tuples.
# Keep one logical thought per cell. Markdown cells are prose; code cells run.
# ---------------------------------------------------------------------------

CELLS = []

def md(text: str) -> None:
    CELLS.append(("markdown", text))

def code(text: str) -> None:
    CELLS.append(("code", text))

# ---------- Title + intro ----------
md("""\
# Units in `bigraph-schema`

`bigraph-schema` ships two complementary mechanisms for working with physical units:

1. **`Quantity`** — the runtime value carries its units alongside its magnitude (backed by [`pint`](https://pint.readthedocs.io/)). You can call `.to(other_unit)`, do unit-safe arithmetic, and inspect `.dimensionality` at runtime.
2. **`Number._units`** — a schema-level annotation. The runtime value is a plain `float`; the unit string lives only on the schema.

A third feature ties them together: when a state's schema declares `_units` and connects to a port that declares a different `_units`, `bigraph-schema` computes a scalar conversion factor at compile time so the value seen through the port is in the port's units.

This notebook walks through each of these.
""")

# ---------- Setup ----------
md("## Setup")
code("""\
from bigraph_schema import allocate_core
core = allocate_core()
""")

# ---------- Section 1: Quantity ----------
md("""\
## 1. `Quantity` — units carried at runtime

Use `'quantity'` when you want unit-aware arithmetic and conversion on the value itself. `realize` accepts several encodings; `serialize` always produces `{units, magnitude}`.
""")

md("### Defining a quantity schema")
code("""\
quantity_schema = core.access({
    '_type': 'quantity',
    'units': {'mol': 1, 'L': -1},
})
quantity_schema
""")

code("""\
core.render(quantity_schema)
""")

md("### Realize from a `{magnitude, units}` dict")
code("""\
_, q, _ = core.realize(
    quantity_schema,
    {'magnitude': 2.0, 'units': {'mol': 1, 'L': -1}},
)
q
""")

code("""\
type(q).__name__, q.magnitude, str(q.units)
""")

md("### Realize from a bare numeric (uses the schema's units)")
code("""\
_, q_bare, _ = core.realize(quantity_schema, 5.0)
q_bare
""")

md("### Realize from a parseable string")
code("""\
_, q_str, _ = core.realize(quantity_schema, '2.1 millimolar')
q_str
""")

md("### Unit-aware operations at runtime")
code("""\
q.to('mmol/L')
""")

code("""\
q.dimensionality
""")

code("""\
(q * 3).to('mol/L')
""")

md("### Serialize roundtrip")
code("""\
encoded = core.serialize(quantity_schema, q)
encoded
""")

code("""\
_, q_back, _ = core.realize(quantity_schema, encoded)
q_back == q
""")

# ---------- Section 2: Number._units ----------
md("""\
## 2. `Number._units` — units as metadata

Use `_units` on a numeric type when you want to record the unit but keep the runtime value as a plain `float` (or numpy array). The unit string lives **only on the schema** — at runtime the value has no awareness of its units.
""")

md("### Defining a unit-annotated float schema")
code("""\
float_schema = core.access({'_type': 'float', '_units': 'mol/L'})
float_schema
""")

code("""\
core.render(float_schema)
""")

md("### The runtime value is a plain `float`")
code("""\
_, x, _ = core.realize(float_schema, 2.5)
x, type(x).__name__
""")

md("### Serialize is just the number")
code("""\
core.serialize(float_schema, x)
""")

md("""\
### When to use which

| Approach        | Runtime value     | Unit info lives in | Use when                                            |
|-----------------|-------------------|--------------------|-----------------------------------------------------|
| `Number._units` | plain `float`     | schema annotation  | hot numerical paths; downstream code expects floats |
| `Quantity`      | `pint.Quantity`   | the value itself   | unit-safe arithmetic / introspection at runtime     |

`_units` is the right default for high-throughput numerics — no per-value pint overhead, and unit-aware wiring (Section 3) still works. Reach for `Quantity` when you need `.to()` or unit-checked arithmetic on individual values.
""")
# ---------- Section 3 placeholder — filled in Task 3 ----------

def cell_dict(cell_type: str, source: str) -> dict:
    src_lines = source.splitlines(keepends=True)
    base = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src_lines,
    }
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base

nb = {
    "cells": [cell_dict(t, s) for t, s in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "units.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
