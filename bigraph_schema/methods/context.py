from bigraph_schema.schema import resolve_path, convert_path
from bigraph_schema.methods.traverse import traverse


@dataclass(kw_only=True)
class Context():
    schema: Node = field(default_factory=Node),
    state: object = field(default_factory=object),
    path: tuple = field(default_factory=tuple)

    def update_path(self, subpath):
        original_path = self.path
        self.path = resolve_path(self.path + subpath)

        return original_path, self.path

    def cursor(self):
        path = convert_path(self.path)
        context = blank_context(self.schema, self.state, path)

        return traverse(self.schema, self.state, path, context)

