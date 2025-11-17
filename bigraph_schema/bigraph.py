"""
attempting a normalized version of the process bigraph abstraction
"""
import uuid

# bigraph{
#     id: uuid
#     nodes : mapping(id:uuid, node:object)
#     places: mapping from id to Place
#     links : mapping(uuid from link, Link)
#     links_nodes : many to many uuid links to nodes}
#
# link{
#     id: uuid
#     redex: callable
#     reactum: callable
#     inner: many to many port_names to node_ids
#     outer: many to many port_names to node_ids
#     schemas: mapping(port_name, schema)}
#
# place{
#    id: uuid
#    inner: list of (place_id or None)
#    outer: single value of (place_id or None)}


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

        inner, outer - many to many mappings from port name to (node_id or None)

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
        # the 'inner' and 'outer' argument keys should each be a list of
        # ('name', schema, node_id), tuples describing one port of the link
        # per item.
        # * 'name' is any identifier, preferring a descriptive string name
        # * schema is a schema that should match the node that gets connected
        # * node_id should be usable to find a node on the parent bigraph, or
        #   None
        # ('name', schema, None) describes an open link.

        inner_ports = kwargs.get('inner', [])
        outer_ports = kwargs.get('outer', [])
        self.inner = JoinDict()
        self.outer = JoinDict()
        # mapping from port name to schema
        self.port_schemas = {}
        for name, schema, node_id in inner_ports:
            self.port_schemas[name] = schema
            node_id = node_id or None
            self.inner.insert(name, node_id)
            bigraph.links_nodes.insert(self.id, node_id)
        for name, schema, node_id in outer_ports:
            self.port_schemas[name] = schema
            node_id = node_id or None
            self.outer.insert(name, node_id)
            bigraph.links_nodes.insert(self.id, node_id)
        bigraph.links[self.id] = self

    def outer_face(self):
        """
        the outer interface of a link is the set of open righthand ports (out
        bound ports not connected to nodes)
        """
        raw = self.outer.get_right(None)
        result = {}
        for name in raw:
            result[name] = {'link_id': self.id,
                            'name': name,
                            'schema': self.port_schemas.get(name)}
        return result

    def inner_face(self):
        """
        the inner interface of a link is the set of open lefthand ports (in
        bound ports not connected to nodes)
        """
        raw = self.inner.get_right(None)
        result = {}
        for name in raw:
            result[name] = {'link_id': self.id,
                            'name': name,
                            'schema': self.port_schemas.get(name)}
        return result

    def process(self, bigraph):
        reactums = self.redex(self, bigraph)
        result = []
        for reactum in reactums:
            result.append(self.reaction(self, bigraph, reactum))
        return result


class Place():
    """
    defines a place in terms of parent (outer) and list of child (inner) nodes
    within a tree, a value of None in either position indicates an interface
    """
    def __init__(self, inner=None, outer=None):
        # list of place_id or None
        self.inner = inner or []
        # single value of place_id or None
        self.outer = outer

    def __repr__(self):
        return f'Place(inner={self.inner}, outer={self.outer})'

    def __eq__(self, other):
        return self.inner == other.inner and self.outer == other.outer

    def as_tree(places):
        roots = Place.get_roots(places)

        result = {}
        for root in roots:
            result[root] = Place.build_tree(places, root)

        return result

    def get_roots(places):
        roots = []
        for node_id,place in places.items():
            if place.outer == None:
                roots.append(node_id)
        return roots

    def build_tree(places, root):
        children = places[root].inner or []
        result = {}
        for child_id in children:
            child = places[child_id]
            if child.inner == None:
                result[child_id] = None
            else:
                result[child_id] = Place.build_tree(places, child_id)
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

        # mapping from node_id to Place
        self.places = {}

        # mapping from link_id to link (might not be needed) if links hold no
        # data
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
        open_links = self.links_nodes.right[None]
        result = {}
        result['links'] = {}
        for link in open_links.keys():
            result['links'][link] = self.links[link].outer_face()

        result['places'] = {}
        for uid,place in self.places.items():
            if place.outer == None:
                result['places'][uid] = place
        return result

    def inner_face(self):
        """
        the inner face is the set of links that join left to None
        """
        open_links = self.links_nodes.right[None]
        result = {}
        result['links'] = {}
        for link in open_links.keys():
            result['links'][link] = self.links[link].inner_face()

        result['places'] = {}
        for uid,place in self.places.items():
            if place.inner == None:
                result['places'][uid] = place
        return result


    def compose(self, interface, other, other_interface):
        """
        based on the links and ports described in interface, and
        other_interface, merge the other bigraph into this one
        """
        pass


    def harvest_tree(self, tree, parent_id, tree_ops):
        """
        navigates `tree' of data using tree_ops:
        `get_children' to find branches,
        `get_leaf' to create nodes,
        and `get_id' to find the id of a node

        `place' should be some subtree of self.places, or a placeholder
        if ingesting nodes without their place structure
        """

        uid = tree_ops.get_id(tree)
        nodes = [uid]
        place = Place(inner=None, outer=parent_id)
        if uid != None:
            self.places[uid] = place
        if parent_id:
            self.places[parent_id].inner.append(uid)

        leaf = tree_ops.get_leaf(tree)
        if leaf != None:
            self.nodes[uid] = leaf
        for k,v in tree_ops.get_children(tree).items():
            new_ids = self.harvest_tree(v, uid, tree_ops)
            nodes.extend(new_ids)
        return nodes


    def as_tree(self, roots=None):
        """
        returns a tree of {'id': uid, 'children': tree, 'data': {}}
        based on root node ids in self.places
        """

        roots = roots or Place.get_roots(self.places)

        result = {}
        for node_id in roots:
            result[node_id] = {'id': node_id}
            node = self.nodes[node_id]
            if node:
                result[node_id]['data'] = node
            place = self.places[node_id]
            if place.inner:
                result[node_id]['children'] = self.as_tree(place.inner)
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


class HarvestTestHelper():
    def get_id(self, node):
        if hasattr(node, 'id'):
            return node.id
        elif node.get('id'):
            return node['id']
        else:
            return None

    def get_children(self, node):
        return node.get('children') or {}

    def get_leaf(self, node):
        return node.get('data')

def test_bigraph():
    bg = Bigraph()

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

    harvested = bg.harvest_tree(branch, None, HarvestTestHelper())
    harvested.sort()

    assert harvested == [1,2,3,4]
    assert Place.as_tree(bg.places) == {
            4: {1: {2: {}},
                3: {}}}
    assert bg.nodes == {
            4: {'a': 0, 'b': 1},
            1: {'b': 0, 'c': 3},
            2: {'d': 1},
            3: {'e': 0}}

    tree = bg.as_tree()

    assert tree == {4: branch}

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
    f_tree = {
            'id': None,
            'children': {
                'v1': {'id': 'v1', 'data': {}},
                'v3': {'id': 'v3', 'data': {}},
                'v4': {'id': 'v4',
                       'data': {},
                       'children': {
                           'v5': {'id': 'v5',
                                  'data': {}}}}}}

    F.harvest_tree( f_tree, None, HarvestTestHelper())


    x_outer = \
            Link(F,
             id='x outer',
             outer=[('v1_port', '*', 'v1'),
                    ('x_port', '*', None)])
    e1 = Link(F,
              id='e1',
              outer=[('v1_port', '*', 'v1'),
                     ('v3_port', '*', 'v3')])
    e2 = Link(F,
              id='e2',
              outer=[('v3_port', '*', 'v3'),
                     ('v4_port', '*', 'v4'),
                     ('v5_port', '*', 'v5')])
    y_outer = \
            Link(F,
             id='y outer',
             outer=[('v4_port', '*','v4'),
                    ('y_port', '*', None)])

    assert F.outer_face() == {
            'links': {
                'x outer': {
                    'x_port': {
                        'link_id': 'x outer',
                        'name': 'x_port',
                        'schema': '*'}},
                'y outer': {
                    'y_port': {
                        'link_id': 'y outer',
                        'name': 'y_port',
                        'schema': '*'}}},
            'places': {
                'v1': Place(inner=[], outer=None),
                'v3': Place(inner=[], outer=None),
                'v4': Place(inner=['v5'], outer=None)}}

    G = Bigraph()
    g_tree = {
            'id': None,
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
                                             'data': {}}}}}},
                'v4': {'id': 'v4',
                       'data': {},
                       'children': {
                           'v5': {'id': 'v5',
                                  'data': {}}}}}}

    G.harvest_tree(g_tree, None, HarvestTestHelper())

    e0 = Link(G,
              id='e0',
              outer=[('v1_port', '*', 'v1'),
                     ('v0_port', '*', 'v0'),
                     ('v4_port', '*', 'v4')])
    e1 = Link(G,
              id='e1',
              outer=[('v1_port', '*', 'v1'),
                     ('v3_port', '*', 'v3')])
    e2 = Link(G,
              id='e2',
              outer=[('v3_port', '*', 'v3'),
                     ('v4_port', '*', 'v4'),
                     ('v5_port', '*', 'v5')])

#    H = Bigraph(nodes=...,
#                links=...,
#                places=[{'parent': None, 'id': 'v0' ...}])
#    H.harvest_tree(
#            {'id': H.id,
#             'data': H,
#             'face': 'inner',
#             'children': {
#                 'v0': {
#                     'id': 'v0',
#                     'data': {},
#                     'children': {
#                         'v2': {
#                             'id': 'v2',
#                             'data': {}}}}}},
#             H.places,
#             HarvestTestHelper())
