"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

import sys
import inspect
import importlib


def function_module(function):
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
