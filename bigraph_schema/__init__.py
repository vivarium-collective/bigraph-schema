from bigraph_schema.registry import (
    deep_merge, validate_merge, default, Registry, hierarchy_depth, is_schema_key, establish_path,
    strip_schema_keys, type_parameter_key, non_schema_keys, set_path, transform_path)

from bigraph_schema.utilities import get_path, visit_method
from bigraph_schema.edge import Edge
from bigraph_schema.type_system import TypeSystem
from bigraph_schema.protocols import local_lookup_module
from bigraph_schema.type_functions import FUNCTION_TYPE, METHOD_TYPE, type_schema_keys, resolve_path

from bigraph_schema.schema import BASE_TYPES
from bigraph_schema.core import Core

import bigraph_schema.methods
