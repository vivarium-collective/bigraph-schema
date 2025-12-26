"""
=========
Protocols
=========

This module contains the protocols for retrieving processes from address.
"""

import sys
import inspect
import importlib


def function_module(function):
    """
    Retrieves the fully qualified name of a given function.
    """
    module = inspect.getmodule(function)

    return f'{module.__name__}.{function.__name__}'


def local_lookup_module(address):
    """Local Module Protocol

    Retrieves local module
    """
    if '.' in address:
        module_name, class_name = address.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        module = sys.modules[__name__]
        if hasattr(module, address):
            return getattr(sys.modules[__name__], address)


def local_lookup_registry(core, address):
    """Process Registry Protocol

    Retrieves from the process registry
    """
    return core.link_registry.get(address)


def local_lookup(core, address):
    """Local Lookup Protocol

    Retrieves local processes, from the process registry or from a local module
    """
    if address[0] == '!':
        instantiate = local_lookup_module(address[1:])
    else:
        instantiate = local_lookup_registry(core, address)
    return instantiate


