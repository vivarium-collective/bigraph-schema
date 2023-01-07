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



class SomeProcess(Process):
    def ports_schema(self):
        return "some_process_schema"

        return {
            "stuff": "field[concentration]"
        }
