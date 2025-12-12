class Places():
    def as_tree(self):
        roots = self.get_roots()

        result = {}
        for root in roots:
            result[root] = self.build_tree(root)

        return result

    def add_branch(self, parent, child):
        pass
    def get_branches(self, node_id):
        pass
    def get_place(self, place_id):
        pass
    def insert(self, place):
        pass
    def get_roots(self):
        pass
    def get_leaves(self):
        pass
    def build_tree(self, root):
        """
        returns a tree of node_id down from root
        """
        pass

class JoinDict():
    """
    many to many mapping between two cachable types
    """
    def __init__(self):
        self.left = {}
        self.right = {}

    def _insert_pair(self, place, a, b):
        if not place.get(a):
            place[a] = {}
        place[a][b] = True

    def insert(self, left, right):
        self._insert_pair(self.left, left, right)
        self._insert_pair(self.right, right, left)

    def _purge_token(self, a, b, token):
        for token_b in a[token].keys():
            b[token_b].pop(token, None)
        a.pop(token, None)

    def remove_left(self, left):
        self._purge_token(self.left, self.right, left)

    def remove_right(self, right):
        self._purge_token(self.right, self.left, right)

    def delink(self, left, right):
        try:
            self.right[right].pop(left)
        except KeyError:
            pass
        try:
            self.left[left].pop(right)
        except KeyError:
            pass

    def get_left(self, item):
        result = self.left.get(item) or {}
        return list(result.keys())

    def get_right(self, item):
        result = self.right.get(item) or {}
        return list(result.keys())

    def update(self, other):
        self.left.update(other.left)
        self.right.update(other.right)

