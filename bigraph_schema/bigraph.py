"""
attempting a normalized version of the process bigraph abstraction
"""

# we implement a "database" of objects keyed by uuid
import uuid

def new_uuid():
    return uuid.uuid4()


class JoinDict():
    "bidirectional mappings between two cachable types"
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

def test_joindict():
    jd = JoinDict()
    jd.insert(0, 'a')
    jd.insert(1, 'a')
    jd.insert(0, 'b')
    jd.insert(2, 'b')

    assert jd.get_left(0) == ['a','b']
    assert jd.get_left(1) == ['a']
    assert jd.get_left(2) == ['b']
    assert jd.get_right('a') == [0,1]
    assert jd.get_right('b') == [0,2]

    jd.remove_left(1)

    assert jd.get_left(0) == ['a','b']
    assert jd.get_left(1) == []
    assert jd.get_left(2) == ['b']
    assert jd.get_right('a') == [0]
    assert jd.get_right('b') == [0,2]

    jd.remove_right('b')

    assert jd.get_left(0) == ['a']
    assert jd.get_left(1) == []
    assert jd.get_left(2) == []
    assert jd.get_right('a') == [0]
    assert jd.get_right('b') == []

    jd.insert(3, 'c')
    jd.insert(3, 'd')
    jd.insert(3, 'e')
    jd.insert(4, 'd')
    jd.delink(3, 'd')

    assert jd.get_left(0) == ['a']
    assert jd.get_right('a') == [0]
    assert jd.get_left(3) == ['c','e']
    assert jd.get_left(4) == ['d']
    assert jd.get_right('c') == [3]
    assert jd.get_right('d') == [4]
    assert jd.get_right('e') == [3]

    # should accept nonsense keys for delink
    assert jd.delink(88, 'z') or True

class Link():
    def __init__(self, **kwargs):
        """

        """
        ## mandatory arguments
        self.node_id = kwargs['node_id']
        self.schema = kwargs['schema']

        ## optional arguments

        self.id = kwargs.get('id') or new_uuid()
        # without a rule and action, we have a passive link
        self.rule = kwargs.get('rule')
        self.action = kwargs.get('action')

        ## registration
        # target takes the form (nod_id, link_id)
        # without target, we have an "open link"
        # we can look up the bigraph's open links via
        # bigraph.links.get_right(None, None)
        target = kwargs.get('target', (None, None))
        bigraph.links[self.id] = self
        # TODO - consider layering links_nodes so it is indexed by node_id?
        bigraph.links_nodes.insert((self.node_id, self.id), target)


class Bigraph():
    """
    an implimentation of a milner process bigraph

    nodes is a mapping from unique id to Object

    the places implements a tree describing place relations, where
    children are "inside" their parent, and the ids can be resolved via
    nodes to get full Object data corresponding to that id

    links describes data connections between two or more ids

    links_nodes is a bimap (two way many to many mapping) from node_id to
    link_ids and from link_id to node ids
    """
    def __init__(self):
        # mapping from node_id to node
        self.nodes = {}

        # tree of uuid
        self.places = {}

        # mapping from link_id to link (might not be needed) if links hold no
        # data
        # weird idea: this could be where schemas belong, or could need its own
        # unified / resolved schema for the link itself
        self.links = {}

        # bidirectional mapping from link_id to node_id
        # links_nodes.left maps link_id to node ids
        # links_nodes.right maps node_id to link_ids
        self.links_nodes = JoinDict()

        self.id = new_uuid()

    # currently no method of importing links is implemented, but it could be
    # done via a get_links helper which expects a relative path(???)
    # also: how to describe open links for merges...
    def harvest_tree(self, tree, place, get_children, get_leaf, get_id):
        """
        navigates `tree' of data using `get_children' to find branches,
        `get_leaf' to create nodes, and `get_id' to find the id of a node

        `place' should be some subtree of self.places, or a placeholder
        if ingesting nodes without their place structure
        """

        uid = get_id(tree)
        nodes = [uid]
        place[uid] = {}

        leaf = get_leaf(tree)
        if leaf:
            self.nodes[uid] = leaf
        for k,v in get_children(tree).items():
            new_ids = self.harvest_tree(v, place[uid],
                                        get_children, get_leaf, get_id)
            nodes.extend(new_ids)
        return nodes

    def graft(self, other, path):
        """
        merges other bigraph into self, nested at `path' into self.places
        """
        self.nodes.update(other.nodes)
        self.links.update(other.links)
        self.links_nodes.upate(other.links_nodes)

        visit = self.places
        for k in path:
            if not visit.get(k):
                visit[k] = {}
            visit = visit[k]
        visit[other.id] = other.places

    # creates a new places such that current places are found
    # below path
    def reparent(self, path):
        """
        creates a nesting around self.places based on `path'
        """
        path = path.copy()
        # walk from innermost step to outermost
        # TODO we can walk inward instead of reversing the list
        path.reverse()

        result = self.places
        for step in path:
            payload = result
            result = {step: payload}

        self.places = result


    def as_tree(self, path=None):
        """
        returns a tree of {'id': uid, 'children': tree, 'data': {}}
        based on self.places starting from `path' spec
        """
        path = path or []

        target = self.places
        for node_id in path:
            step = target[node_id]
            target = step

        target_id = None
        target_data = None
        if len(path) > 0:
            target_id = path[-1]
            target_data = self.nodes.get(target_id)
        else:
            target_id = self.id
            target_data = self

        children = {}
        for k,v in target.items():
            childpath = path.copy()
            childpath.append(k)
            children[k] = self.as_tree(path=childpath)

        result = {
                'id': target_id,
                'data': target_data}
        if len(children) > 0:
            result['children'] = children
        return result


def test_bigraph():
    bg = Bigraph()
    tree = bg.as_tree()
    assert tree == {
            'id': bg.id,
            'data': bg}

    def get_children(node):
        return node.get('children') or {}
    def get_leaf(node):
        return node.get('data') or {}
    def get_id(node):
        if hasattr(node, 'id'):
            return node.id
        elif node.get('id'):
            return node['id']
        else:
            return new_uuid()

    branch = {
            'id': 4,
            'data': {'a': 0, 'b': 1},
            'children': {
                1: {'id': 1,
                    'data': {'b': 0, 'c': 3},
                    'children': {
                        2: {'id': 2,
                            'data': {'d': 1}}}},
                3: {'id': 3,
                    'data': {'e': 0}}}}

    harvested = bg.harvest_tree(branch, bg.places, \
            get_children, get_leaf, get_id)

    harvested.sort()
    assert harvested == [1,2,3,4]
    assert bg.places == {
            4: {1: {2: {}},
                3: {}}}
    assert bg.nodes == {
            4: {'a': 0, 'b': 1},
            1: {'b': 0, 'c': 3},
            2: {'d': 1},
            3: {'e': 0}}

    tree = bg.as_tree()

    assert tree == {
            'id': bg.id,
            'data': bg,
            'children': {4: branch}}

    bg2 = Bigraph()
    branch2  {
            'children': {
                5: {'id': 5,
                    'data': {'f': 4}},
                5: {'id': 6,
                    'data': {'g': 4}}}}
