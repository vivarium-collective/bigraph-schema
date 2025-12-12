from bigraph_schema.bigraph.basis import Places, JoinDict

# class Placement():
#     """
#     """
#
#     def __init__(self, outer, inner):
#         self.outer = outer
#         self.inner = inner
#
#     def __repr__(self):
#         return f"Placement({self.outer}, {self.inner})"
#
#     def __eq__(self, other):
#         return (self.inner, self.outer) == (other.inner, other.outer)


class PlacementPlaces():
    """
    """

    def __init__(self):
        self.places = JoinDict

    def insert(self, (outer, inner)):
        self.places.insert(outer, inner)

    def add_branch(self, parent, child):
        self.insert((parent, child))

    def get_place(self, node_id):
        [parent] = self.places.get_right(node_id)
        children = self.places.get_left(node_id)
        return {'outer': parent, 'inner': children}

    def get_branches(self, node_id):
        assert False

    def get_roots(self):
        return self.places.get_left(None)

    def get_leaves(self):
        return self.places.get_right(None)

    def build_tree(self, root):
        branches = self.places.get_left(root)
        # this case indicates a port
        if branches == [None]:
            return None
        result = {}
        for branch in branches:
            result[branch] = self.build_tree(branch)
        return result
