"""
attempting a normalized version of the process bigraph abstraction
"""
import uuid

# bigraph{
#     id: uuid
#     nodes : mapping(id:uuid, node:object)
#     places : tree(uuid from nodes)
#     links : mapping(uuid from link, Link)
#     links_nodes : many to many uuid links to nodes}
#
# link{
#     id: uuid
#     rule: callable
#     action: callable
#     ports: many to many port_names to node_ids
#     schemas: mapping(port_name, schema)}


def new_uuid():
    return uuid.uuid4()


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


class Link():
    def __init__(self, bigraph, **kwargs):
        """
        creates an object with the following properties:

        id - provided by kwargs or generated as a uuid, used to register
            link to the bigraph, and register connections to nodes in the
            bigraph

        redex, reaction - used together to update bigraph

        ports - a many to many mapping from port name to (node_id or None)

        port_schemas - mapping from port name to schema

        updates bigraph.links and bigraph.links_nodes as apropriate
        """

        self.id = kwargs.get('id') or new_uuid()

        # self.redex takes (self, bigraph), and returns a series of
        # 0 or more argument objects for calls to self.reaction
        # should not mutate bigraph
        self.redex = kwargs.get('redex')
        # self.reaction takes (self, bigraph, arglist) and returns a result
        # object
        # should mutate bigraph
        self.reaction = kwargs.get('reaction')

        ## registration
        # the 'ports' argument key should be a list of
        # ('name', schema, node_id), tuples describing one port of the link
        # per item.
        # * 'name' is any identifier, preferring a descriptive string name
        # * schema is a schema that should match the node that gets connected
        # * node_id should be usable to find a node on the parent bigraph, or
        #   None
        # ('name', schema, None) describes an open link.

        raw_ports = kwargs.get('ports', [])
        self.ports = JoinDict()
        # mapping from port name to schema
        self.port_schemas = {}
        for name, schema, node_id in raw_ports:
            self.port_schemas[name] = schema
            node_id = node_id or None
            self.ports.insert(name, node_id)
            bigraph.links_nodes.insert(self.id, node_id)
        bigraph.links[self.id] = self

    def outer_face(self):
        """
        the outer interface of a link is the set of open righthand ports (out
        bound ports not connected to nodes)
        """
        raw = self.ports.get_right(None)
        result = {}
        for name in raw:
            result[name] = {'link_id': self.id,
                            'name': name,
                            'schema': self.port_schemas[name]}
        return result

    def inner_face(self):
        """
        the inner interface of a link is the set of open lefthand ports (in
        bound ports not connected to nodes)
        """
        pass

    def process(self, bigraph):
        reactums = self.redex(self, bigraph)
        result = []
        for reactum in reactums:
            result.append(self.reaction(self, bigraph, reactum))
        return result


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

    def outer_face(self):
        """
        the outer face is the set of links that join right to None
        """
        outer = self.links_nodes.right[None]
        result = {}
        for link in outer.keys():
            result[link] = self.links[link].outer_face()
        return result

    def inner_face(self):
        """
        the inner face is the set of links that join left to None
        """
        inner = self.links_nodes.left[None]
        result = []
        for link in inner.keys():
            result.append(self.links[link].inner_face())
        return result


    # TODO: inner vs. outer interface
    # this will need to use the Links class defined above
    def compose(self, interface, other, other_interface):
        """
        based on the links and ports described in interface, and
        other_interface, merge the other bigraph into this one
        """
        pass

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
        if leaf != None:
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


def get_id(node):
    if hasattr(node, 'id'):
        return node.id
    elif node.get('id'):
        return node['id']
    else:
        return new_uuid()
# harvest_tree test helpers
harvest_tree_helpers = {
        'get_children': lambda node: node.get('children') or {},
        'get_leaf': lambda node: node.get('data') or {},
        'get_id': get_id}
del get_id

def test_bigraph():
    bg = Bigraph()
    tree = bg.as_tree()
    assert tree == {
            'id': bg.id,
            'data': bg}


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
            harvest_tree_helpers['get_children'],
            harvest_tree_helpers['get_leaf'],
            harvest_tree_helpers['get_id'])

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
    branch2  = {
            'children': {
                5: {'id': 5,
                    'data': {'f': 4}},
                5: {'id': 6,
                    'data': {'g': 4}}}}

def test_link():
    pass

def test_link_bigraph():
    """
    based on example of linking bigraphs in chapter 1 of the milner book,
    F composed with H should result in G
    """
    F = Bigraph()
    F.harvest_tree(
            # TODO - should the bigraph id be on the root?
            # should ids for root nodes be valid without the
            # 'children' tag at the top level?
            {'id': F.id,
             'data': F,
             'children': {
                 'v1': {'id': 'v1', 'data': {}},
                 'v3': {'id': 'v3', 'data': {}},
                 'v4': {'id': 'v4',
                        'data': {},
                        'children': {
                            'v5': {'id': 'v5',
                                   'data': {}}}}}},
        F.places,
        harvest_tree_helpers['get_children'],
        harvest_tree_helpers['get_leaf'],
        harvest_tree_helpers['get_id'])


    x_outer = \
            Link(F,
             id='x outer',
             ports=[('v1_port', '*', 'v1'),
                    ('x_port', '*', None)])
    e1 = Link(F,
              id='e1',
              ports=[('v1_port', '*', 'v1'),
                     ('v3_port', '*', 'v3')])
    e2 = Link(F,
              id='e2',
              ports=[('v3_port', '*', 'v3'),
                     ('v4_port', '*', 'v4'),
                     ('v5_port', '*', 'v5')])
    y_outer = \
            Link(F,
             id='y outer',
             ports=[('v4_port', '*','v4'),
                    ('y_port', '*', None)])

    assert F.outer_face() == {
            'x outer': {
                'x_port': {
                    'link_id': 'x outer',
                    'name': 'x_port',
                    'schema': '*'}},
            'y outer': {
                'y_port': {
                    'link_id': 'y outer',
                    'name': 'y_port',
                    'schema': '*'}}}

    G = Bigraph()
    G.harvest_tree(
            {'id': G.id,
             'data': G,
             'children': {
                 'v0': {'id': 'v0',
                        'data': {},
                        'children': {
                            'v1': {'id': 'v1',
                                   'data': {}},
                            'v2': {'id': 'v2',
                                   'data': {},
                                   'children': {
                                       'v3': {'id': 'v3',
                                              'data': {}}}}}}},
                 'v4': {'id': 'v4',
                        'data': {},
                        'children': {
                            'v5': {'id': 'v5',
                                   'data': {}}}}},
                G.places,
                harvest_tree_helpers['get_children'],
                harvest_tree_helpers['get_leaf'],
                harvest_tree_helpers['get_id'])

    e0 = Link(G,
              id='e0',
              ports=[('v1_port', '*', 'v1'),
                     ('v0_port', '*', 'v0'),
                     ('v4_port', '*', 'v4')])
    e1 = Link(G,
              id='e1',
              ports=[('v1_port', '*', 'v1'),
                     ('v3_port', '*', 'v3')])
    e2 = Link(G,
              id='e2',
              ports=[('v3_port', '*', 'v3'),
                     ('v4_port', '*', 'v4'),
                     ('v5_port', '*', 'v5')])

    H = Bigraph()
    H.harvest_tree(
            {'id': H.id,
             'data': H,
             'children': {
                 'v0': {
                     'id': 'v0',
                     'data': {},
                     'children': {
                         'v2': {
                             'id': 'v2',
                             'data': {}}}}}},
             H.places,
             harvest_tree_helpers['get_children'],
             harvest_tree_helpers['get_leaf'],
             harvest_tree_helpers['get_id'])

    # x_inner = \
    #        Link(H,
    #             id='x inner',
    #             ports=[(None, '*', 'x_port')]
