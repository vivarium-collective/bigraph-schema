# some types are given
# * bool
# * int
# * float
# * string
# * tuple
# * array
# * dict
# * units

parse("m / s") --> m / s

cell_schema = {
    "transporters": {
        "_type": "branch[string, concentration]"
    },

    "transporters": {
        "*": {
            "_type": "concentration"
        }
    },

    "exchange": {
        "glucose": "counts",
        "lactose": "counts",
        "galactose": "counts",
    },
}

environment_schema = {
    "field": {
        "glucose": "field[concentration]",
        "lactose": "field[concentration]",
        "galactose": "field[concentration]",
    },

    "cells": "branch[string, cell_data]"
}

schema_registry = {
    "location": "tuple[float, float]",
    "counts": "int",
    "concentration": {
        "magnitude": "float",
        "units": "units"},
    "field[A]": {
        "array[array[A]]"},

    "cell_data": {
        "location": "location",
        "id": "int"},
    "cell": cell_schema,
    "environment": environment_schema,
}

def compose_environment_cell():
    # figure this out
    cell_in_environment_schema = compose_schemas({
        "environment": environment_schema,
        "cell": cell_schema}, {
        "environment": {},
        "cell": {}})
    print(cell_in_environment_schema)

def main():
    compose_environment_cell()

if __name__ == "__main__":
    main()
