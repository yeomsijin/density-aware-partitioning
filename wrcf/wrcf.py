# Derived from kLabUM/rrcf (MIT); see docs/licenses/rrcf.txt.
# Weighted batch cuts: Sijin Yeom. Modern API and rebuild updates: 2026.
from numbers import Integral

import numpy as np

from dap.partition import density_aware_split


class RCTree:
    """Weighted random cut tree with range-weighted features and count < alpha cuts.

    Parameters
    ----------
    X : finite array of shape (n_samples, n_features), optional
        None creates an empty tree. Singleton and duplicate rows are supported.
    index_labels : sequence of unique hashable labels, optional
    precision : nonnegative int or None, default=None
        Decimal rounding applied to both initial and inserted observations.
    random_state : int, Generator, RandomState or None
    alpha : int >= 2, default=2
        Weighted-paper tolerance: accept window count < alpha.
    update_mode : {'rebuild', 'legacy'}, default='rebuild'
        Rebuild keeps all current cuts weighted; legacy insertion is uniform RRCF.

    The underlying tree/CODISP data structure derives from kLabUM/rrcf (MIT).
    See docs/API.md for migration and serialization semantics.
    """

    def __init__(
        self,
        X=None,
        index_labels=None,
        precision=None,
        random_state=None,
        alpha=2,
        update_mode="rebuild",
    ):
        if isinstance(alpha, bool) or not isinstance(alpha, Integral) or alpha < 2:
            raise ValueError("alpha must be an integer >= 2 (accept count < alpha)")
        if update_mode not in ("rebuild", "legacy"):
            raise ValueError("update_mode must be 'rebuild' or 'legacy'")
        if precision is not None and (
            isinstance(precision, bool) or not isinstance(precision, Integral) or precision < 0
        ):
            raise ValueError("precision must be None or a nonnegative integer")
        self.alpha, self.precision, self.update_mode = alpha, precision, update_mode
        self.rng = (
            random_state
            if isinstance(random_state, (np.random.RandomState, np.random.Generator))
            else np.random.default_rng(random_state)
        )
        self.leaves, self.root, self.ndim = {}, None, None
        if X is None:
            return
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array")
        if precision is not None:
            X = np.around(X, decimals=precision)
        if not np.isfinite(np.ptp(X, axis=0)).all():
            raise ValueError("feature range overflows; rescale X")
        labels = list(range(len(X))) if index_labels is None else list(index_labels)
        if len(labels) != len(X) or len(set(labels)) != len(labels):
            raise ValueError("index_labels must contain one unique hashable label per row")
        self.index_labels = labels
        U, I, N = np.unique(X, return_inverse=True, return_counts=True, axis=0)
        self.ndim = X.shape[1]
        self.u = None
        self._multiplicities = N
        if len(U) == 1:
            self.root = Leaf(i=labels[0], d=0, x=U[0].copy(), n=int(N[0]))
            self.leaves = dict.fromkeys(labels, self.root)
        else:
            self._mktree(U, np.ones(len(U), dtype=bool), N, I, parent=self)
            self.root.u = None
            self._count_all_top_down(self.root)
            self._get_bbox_top_down(self.root)
        del self._multiplicities

    def _rebuild(self, rows, labels):
        self.__init__(
            np.asarray(rows) if rows else None,
            index_labels=labels,
            precision=self.precision,
            random_state=self.rng,
            alpha=self.alpha,
            update_mode=self.update_mode,
        )

    def insert_point(self, point, index, tolerance=None):
        """Insert and fully rebuild weighted cuts by default.

        Rebuild is an explicit engineering extension, not a claim of a derived
        efficient online WRCF law. update_mode='legacy' retains the old RRCF
        insertion (uniform, NOT density-aware). Rebuild invalidates old Leaf refs.
        """
        point = np.asarray(point, dtype=float)
        if point.ndim != 1 or not point.size or not np.isfinite(point).all():
            raise ValueError("point must be a nonempty finite 1D array")
        if self.ndim is not None and point.size != self.ndim:
            raise ValueError("point has the wrong number of features")
        if index in self.leaves:
            raise KeyError("index already exists")
        if self.precision is not None:
            point = np.around(point, decimals=self.precision)
        if self.update_mode == "legacy":
            return self._insert_point_legacy(point, index, tolerance=tolerance)
        if tolerance is not None:
            raise ValueError(
                "tolerance is supported only in legacy update mode; use precision for rebuild"
            )
        labels = list(self.leaves)
        rows = [self.leaves[label].x.copy() for label in labels]
        self._rebuild(rows + [point.copy()], labels + [index])
        return self.leaves[index]

    def forget_point(self, index):
        """Remove an observation; weighted rebuild by default, including duplicates."""
        if self.update_mode == "legacy":
            return self._forget_point_legacy(index)
        leaf = self.leaves[index]
        labels = [label for label in self.leaves if label != index]
        self._rebuild([self.leaves[label].x.copy() for label in labels], labels)
        return leaf

    def __repr__(self):
        depth = ""
        treestr = ""

        def print_push(char):
            nonlocal depth
            branch_str = " {}  ".format(char)
            depth += branch_str

        def print_pop():
            nonlocal depth
            depth = depth[:-4]

        def print_tree(node):
            nonlocal depth
            nonlocal treestr
            if isinstance(node, Leaf):
                treestr += "({})\n".format(node.i)
            elif isinstance(node, Branch):
                treestr += "{0}{1}\n".format(chr(9472), "+")
                treestr += "{0} {1}{2}{2}".format(depth, chr(9500), chr(9472))
                print_push(chr(9474))
                print_tree(node.l)
                print_pop()
                treestr += "{0} {1}{2}{2}".format(depth, chr(9492), chr(9472))
                print_push(" ")
                print_tree(node.r)
                print_pop()

        print_tree(self.root)
        return treestr

    def _cut(self, X, S, parent=None, side="l"):
        # Find max and min over all d dimensions
        xmax = X[S].max(axis=0)
        xmin = X[S].min(axis=0)
        # Compute l
        l = xmax - xmin
        l /= l.sum()
        # Determine dimension to cut
        q = self.rng.choice(self.ndim, p=l)
        # Retain full observation multiplicity even when duplicate rows share leaves.
        y = np.repeat(X[S, q], self._multiplicities[S])
        p = density_aware_split(y, alpha=self.alpha - 1, random_state=self.rng)
        # RCF routes <= to the left; ensure the maximum is not selected by rounding.
        p = min(p, float(np.nextafter(xmax[q], xmin[q])))
        # Determine subset of points to left
        S1 = (X[:, q] <= p) & (S)
        # Determine subset of points to right
        S2 = (~S1) & (S)
        # Create new child node
        child = Branch(q=q, p=p, u=parent)
        # Link child node to parent
        if parent is not None:
            setattr(parent, side, child)
        return S1, S2, child

    def _mktree(self, X, S, N, I, parent=None, side="root", depth=0):
        # Increment depth as we traverse down
        depth += 1
        # Create a cut according to definition 1
        S1, S2, branch = self._cut(X, S, parent=parent, side=side)
        # If S1 does not contain an isolated point...
        if S1.sum() > 1:
            # Recursively construct tree on S1
            self._mktree(X, S1, N, I, parent=branch, side="l", depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.flatnonzero(S1).item()
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :], n=N[i])
            # Link leaf node to parent
            branch.l = leaf
            # If duplicates exist...
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices
                J = np.flatnonzero(I == i)
                # Get index label
                J = [self.index_labels[j] for j in J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # If S2 does not contain an isolated point...
        if S2.sum() > 1:
            # Recursively construct tree on S2
            self._mktree(X, S2, N, I, parent=branch, side="r", depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.flatnonzero(S2).item()
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :], n=N[i])
            # Link leaf node to parent
            branch.r = leaf
            # If duplicates exist...
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices
                J = np.flatnonzero(I == i)
                # Get index label
                J = [self.index_labels[j] for j in J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # Decrement depth as we traverse back up
        depth -= 1

    def map_leaves(self, node, op=(lambda x: None), *args, **kwargs):
        """
        Traverse tree recursively, calling operation given by op on leaves

        Parameters:
        -----------
        node: node in RCTree
        op: function to call on each leaf
        *args: positional arguments to op
        **kwargs: keyword arguments to op

        Returns:
        --------
        None

        Example:
        --------
        # Use map_leaves to print leaves in postorder
        >>> X = np.random.randn(10, 2)
        >>> tree = RCTree(X)
        >>> tree.map_leaves(tree.root, op=print)

        Leaf(5)
        Leaf(9)
        Leaf(4)
        Leaf(0)
        Leaf(6)
        Leaf(2)
        Leaf(3)
        Leaf(7)
        Leaf(1)
        Leaf(8)
        """
        if isinstance(node, Branch):
            if node.l:
                self.map_leaves(node.l, op=op, *args, **kwargs)
            if node.r:
                self.map_leaves(node.r, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def map_branches(self, node, op=(lambda x: None), *args, **kwargs):
        """
        Traverse tree recursively, calling operation given by op on branches

        Parameters:
        -----------
        node: node in RCTree
        op: function to call on each branch
        *args: positional arguments to op
        **kwargs: keyword arguments to op

        Returns:
        --------
        None

        Example:
        --------
        # Use map_branches to collect all branches in a list
        >>> X = np.random.randn(10, 2)
        >>> tree = RCTree(X)
        >>> branches = []
        >>> tree.map_branches(tree.root, op=(lambda x, stack: stack.append(x)),
                            stack=branches)
        >>> branches

        [Branch(q=0, p=-0.53),
        Branch(q=0, p=-0.35),
        Branch(q=1, p=-0.67),
        Branch(q=0, p=-0.15),
        Branch(q=0, p=0.23),
        Branch(q=1, p=0.29),
        Branch(q=1, p=1.31),
        Branch(q=0, p=0.62),
        Branch(q=1, p=0.86)]
        """
        if isinstance(node, Branch):
            if node.l:
                self.map_branches(node.l, op=op, *args, **kwargs)
            if node.r:
                self.map_branches(node.r, op=op, *args, **kwargs)
            op(node, *args, **kwargs)

    def _forget_point_legacy(self, index):
        """
        Delete leaf from tree

        Parameters:
        -----------
        index: (Hashable type)
               Index of leaf in tree

        Returns:
        --------
        leaf: Leaf instance
              Deleted leaf

        Example:
        --------
        # Create RCTree
        >>> tree = RCTree()

        # Insert a point
        >>> x = np.random.randn(2)
        >>> tree.insert_point(x, index=0)

        # Forget point
        >>> tree.forget_point(0)
        """
        try:
            # Get leaf from leaves dict
            leaf = self.leaves[index]
        except KeyError:
            raise KeyError("Leaf must be a key to self.leaves")
        # If duplicate points exist...
        if leaf.n > 1:
            # Simply decrement the number of points in the leaf and for all branches above
            self._update_leaf_count_upwards(leaf, inc=-1)
            return self.leaves.pop(index)
        # Weird cases here:
        # If leaf is the root...
        if leaf is self.root:
            self.root = None
            self.ndim = None
            return self.leaves.pop(index)
        # Find parent
        parent = leaf.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # If parent is the root...
        if parent is self.root:
            # Delete parent
            del parent
            # Set sibling as new root
            sibling.u = None
            self.root = sibling
            # Update depths
            if isinstance(sibling, Leaf):
                sibling.d = 0
            else:
                self.map_leaves(sibling, op=self._increment_depth, inc=-1)
            return self.leaves.pop(index)
        # Find grandparent
        grandparent = parent.u
        # Set parent of sibling to grandparent
        sibling.u = grandparent
        # Short-circuit grandparent to sibling
        if parent is grandparent.l:
            grandparent.l = sibling
        else:
            grandparent.r = sibling
        # Update depths
        parent = grandparent
        self.map_leaves(sibling, op=self._increment_depth, inc=-1)
        # Update leaf counts under each branch
        self._update_leaf_count_upwards(parent, inc=-1)
        # Update bounding boxes
        point = leaf.x
        self._relax_bbox_upwards(parent, point)
        return self.leaves.pop(index)

    def _update_leaf_count_upwards(self, node, inc=1):
        """
        Called after inserting or removing leaves. Updates the stored count of leaves
        beneath each branch (branch.n).
        """
        while node:
            node.n += inc
            node = node.u

    def _insert_point_legacy(self, point, index, tolerance=None):
        """
        Inserts a point into the tree, creating a new leaf

        Parameters:
        -----------
        point: np.ndarray (1 x d)
        index: (Hashable type)
               Identifier for new leaf in tree
        tolerance: float
                   Tolerance for determining duplicate points

        Returns:
        --------
        leaf: Leaf
              New leaf in tree

        Example:
        --------
        # Create RCTree
        >>> tree = RCTree()

        # Insert a point
        >>> x = np.random.randn(2)
        >>> tree.insert_point(x, index=0)
        """
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if self.root is None:
            leaf = Leaf(x=point, i=index, d=0)
            self.root = leaf
            self.ndim = point.size
            self.leaves[index] = leaf
            return leaf
        # If leaves already exist in tree, check dimensions of point
        try:
            assert point.size == self.ndim
        except AssertionError:
            raise ValueError("Point must be same dimension as existing points in tree.")
        # Check for existing index in leaves dict
        try:
            assert index not in self.leaves
        except AssertionError:
            raise KeyError("Index already exists in leaves dict.")
        # Check for duplicate points
        duplicate = self.find_duplicate(point, tolerance=tolerance)
        if duplicate:
            self._update_leaf_count_upwards(duplicate, inc=1)
            self.leaves[index] = duplicate
            return duplicate
        # If tree has points and point is not a duplicate, continue with main algorithm...
        node = self.root
        parent = node.u
        maxdepth = max([leaf.d for leaf in self.leaves.values()])
        depth = 0
        branch = None
        for _ in range(maxdepth + 1):
            bbox = node.b
            cut_dimension, cut = self._insert_point_cut(point, bbox)
            if cut <= bbox[0, cut_dimension]:
                leaf = Leaf(x=point, i=index, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=leaf, r=node, n=(leaf.n + node.n))
                break
            elif cut >= bbox[-1, cut_dimension]:
                leaf = Leaf(x=point, i=index, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=node, r=leaf, n=(leaf.n + node.n))
                break
            else:
                depth += 1
                if point[node.q] <= node.p:
                    parent = node
                    node = node.l
                    side = "l"
                else:
                    parent = node
                    node = node.r
                    side = "r"
        try:
            assert branch is not None
        except AssertionError:
            raise AssertionError("Error with program logic: a cut was not found.")
        # Set parent of new leaf and old branch
        node.u = branch
        leaf.u = branch
        # Set parent of new branch
        branch.u = parent
        if parent is not None:
            # Set child of parent to new branch
            setattr(parent, side, branch)
        else:
            # If a new root was created, assign the attribute
            self.root = branch
        # Increment depths below branch
        self.map_leaves(branch, op=self._increment_depth, inc=1)
        # Increment leaf count above branch
        self._update_leaf_count_upwards(parent, inc=1)
        # Update bounding boxes
        self._tighten_bbox_upwards(branch)
        # Add leaf to leaves dict
        self.leaves[index] = leaf
        # Return inserted leaf for convenience
        return leaf

    def query(self, point, node=None):
        """
        Search for leaf nearest to point

        Parameters:
        -----------
        point: np.ndarray (1 x d)
               Point to search for
        node: Branch instance
              Defaults to root node

        Returns:
        --------
        nearest: Leaf
                 Leaf nearest to queried point in the tree

        Example:
        --------
        # Create RCTree
        >>> X = np.random.randn(10, 2)
        >>> tree = rrcf.RCTree(X)

        # Insert new point
        >>> new_point = np.array([4, 4])
        >>> tree.insert_point(new_point, index=10)

        # Query tree for point with added noise
        >>> tree.query(new_point + 1e-5)

        Leaf(10)
        """
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if node is None:
            node = self.root
        return self._query(point, node)

    def disp(self, leaf):
        """
        Compute displacement at leaf

        Parameters:
        -----------
        leaf: index of leaf or Leaf instance

        Returns:
        --------
        displacement: int
                      Displacement if leaf is removed

        Example:
        --------
        # Create RCTree
        >>> X = np.random.randn(100, 2)
        >>> tree = rrcf.RCTree(X)
        >>> new_point = np.array([4, 4])
        >>> tree.insert_point(new_point, index=100)

        # Compute displacement
        >>> tree.disp(100)

        12
        """
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError("leaf must be a Leaf instance or key to self.leaves")
        # Handle case where leaf is root
        if leaf is self.root:
            return 0
        parent = leaf.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # Count number of nodes in sibling subtree
        displacement = sibling.n
        return displacement

    def codisp(self, leaf):
        """
        Compute collusive displacement at leaf

        Parameters:
        -----------
        leaf: index of leaf or Leaf instance

        Returns:
        --------
        codisplacement: float
                        Collusive displacement if leaf is removed.

        Example:
        --------
        # Create RCTree
        >>> X = np.random.randn(100, 2)
        >>> tree = rrcf.RCTree(X)
        >>> new_point = np.array([4, 4])
        >>> tree.insert_point(new_point, index=100)

        # Compute collusive displacement
        >>> tree.codisp(100)

        31.667
        """
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError("leaf must be a Leaf instance or key to self.leaves")
        # Handle case where leaf is root
        if leaf is self.root:
            return 0
        node = leaf
        results = []
        for _ in range(node.d):
            parent = node.u
            if parent is None:
                break
            if node is parent.l:
                sibling = parent.r
            else:
                sibling = parent.l
            num_deleted = node.n
            displacement = sibling.n
            result = displacement / num_deleted
            results.append(result)
            node = parent
        co_displacement = max(results)
        return co_displacement

    def get_bbox(self, branch=None):
        """
        Compute bounding box of all points underneath a given branch.

        Parameters:
        -----------
        branch: Branch instance
                Starting branch. Defaults to root of tree.

        Returns:
        --------
        bbox: np.ndarray (2 x d)
              Bounding box of all points underneath branch

        Example:
        --------
        # Create RCTree and compute bbox
        >>> X = np.random.randn(10, 3)
        >>> tree = rrcf.RCTree(X)
        >>> tree.get_bbox()

        array([[-0.8600458 , -1.69756215, -1.16659065],
               [ 2.48455863,  1.02869042,  1.09414144]])
        """
        if branch is None:
            branch = self.root
        mins = np.full(self.ndim, np.inf)
        maxes = np.full(self.ndim, -np.inf)
        self.map_leaves(branch, op=self._get_bbox, mins=mins, maxes=maxes)
        bbox = np.vstack([mins, maxes])
        return bbox

    def find_duplicate(self, point, tolerance=None):
        """
        If point is a duplicate of existing point in the tree, return the leaf
        containing the point, else return None.

        Parameters:
        -----------
        point: np.ndarray (1 x d)
               Point to query in the tree.

        tolerance: float
                   Tolerance for determining whether or not point is a duplicate.

        Returns:
        --------
        duplicate: Leaf or None
                   If point is a duplicate, returns the leaf containing the point.
                   If point is not a duplicate, return None.

        Example:
        --------
        # Create RCTree
        >>> X = np.random.randn(10, 2)
        >>> tree = rrcf.RCTree(X)

        # Insert new point
        >>> new_point = np.array([4, 4])
        >>> tree.insert_point(new_point, index=10)

        # Search for duplicates
        >>> tree.find_duplicate((3, 3))

        >>> tree.find_duplicate((4, 4))

        Leaf(10)
        """
        if self.root is None:
            return None
        nearest = self.query(point)
        if tolerance is None:
            if (nearest.x == point).all():
                return nearest
        else:
            if np.isclose(nearest.x, point, rtol=tolerance).all():
                return nearest
        return None

    def to_dict(self):
        """
        Serializes RCTree to a nested dict that can be written to disk or sent
        over a network (e.g. as json).

        Returns:
        --------
        obj: dict
             Nested dictionary representing all nodes in the RCTree.

        Example:
        --------
        # Create RCTree
        >>> X = np.random.randn(4, 3)
        >>> tree = rrcf.RCTree(X)

        # Write tree to dict
        >>> obj = tree.to_dict()
        >>> print(obj)

        # Write dict to file
        >>> import json
        >>> with open('tree.json', 'w') as outfile:
                json.dump(obj, outfile)
        """
        # Create empty dict
        obj = {}
        # Create dict to keep track of duplicates
        duplicates = {}
        for k, v in self.leaves.items():
            if isinstance(k, np.int64):
                duplicates.setdefault(v, []).append(int(k))
            else:
                duplicates.setdefault(v, []).append(k)
        # Serialize tree to dict
        if self.root is None:
            obj["type"] = "Empty"
        else:
            self._serialize(self.root, obj, duplicates)
        obj["_wrcf"] = {
            "alpha": self.alpha,
            "precision": self.precision,
            "update_mode": self.update_mode,
        }
        return obj

    def _serialize(self, node, obj, duplicates):
        """
        Recursively serializes tree into a nested dict.
        """
        if isinstance(node, Branch):
            obj["type"] = "Branch"
            obj["q"] = int(node.q)
            obj["p"] = float(node.p)
            obj["n"] = int(node.n)
            obj["b"] = node.b.tolist()
            obj["l"] = {}
            obj["r"] = {}
            if node.l:
                self._serialize(node.l, obj["l"], duplicates)
            if node.r:
                self._serialize(node.r, obj["r"], duplicates)
        elif isinstance(node, Leaf):
            if isinstance(node.i, np.int64):
                i = int(node.i)
            else:
                i = node.i
            obj["type"] = "Leaf"
            obj["i"] = i
            obj["x"] = node.x.tolist()
            obj["d"] = int(node.d)
            obj["n"] = int(node.n)
            obj["ixs"] = duplicates[node]
        else:
            raise TypeError("`node` must be Branch or Leaf instance")

    def load_dict(self, obj):
        """
        Deserializes a nested dict representing an RCTree and loads into the RCTree
        instance. Note that this will delete all data in the current RCTree and
        replace it with the loaded data.

        Parameters:
        -----------
        obj: dict
             Nested dictionary representing all nodes in the RCTree.

        Example:
        --------
        # Load dict (see to_dict method for more info)
        >>> import json
        >>> with open('tree.json', 'r') as infile:
                obj = json.load(infile)

        # Create empty RCTree and load data
        >>> tree = rrcf.RCTree()
        >>> tree.load_dict(obj)

        # View loaded data
        >>> print(tree)
        >>>
        ─+
        ├───+
        │   ├──(3)
        │   └───+
        │       ├──(2)
        │       └──(0)
        └──(1)
        """
        config = obj.get("_wrcf", {})
        self.__init__(
            random_state=self.rng,
            alpha=config.get("alpha", self.alpha),
            precision=config.get("precision", self.precision),
            update_mode=config.get("update_mode", self.update_mode),
        )
        if obj.get("type") == "Empty":
            return
        # Create anchor node
        anchor = Branch(q=None, p=None)
        # Create dictionary for restoring duplicates
        duplicates = {}
        # Deserialize json object
        self._deserialize(obj, anchor, duplicates)
        # Get root node
        root = anchor.l
        root.u = None
        # Fill in leaves dict
        leaves = {}
        for k, v in duplicates.items():
            for i in v:
                leaves[i] = k
        # Set root of tree to new root
        self.root = root
        self.leaves = leaves
        # Set number of dimensions based on first leaf
        self.ndim = len(next(iter(leaves.values())).x)

    def _deserialize(self, obj, node, duplicates, side="l"):
        """
        Recursively deserializes tree from a nested dict.
        """
        if obj["type"] == "Branch":
            q = obj["q"]
            p = obj["p"]
            n = np.int64(obj["n"])
            b = np.asarray(obj["b"])
            branch = Branch(q=q, p=p, n=n, b=b, u=node)
            setattr(node, side, branch)
            if "l" in obj:
                self._deserialize(obj["l"], branch, duplicates, side="l")
            if "r" in obj:
                self._deserialize(obj["r"], branch, duplicates, side="r")
        elif obj["type"] == "Leaf":
            i = obj["i"]
            x = np.asarray(obj["x"])
            d = obj["d"]
            n = np.int64(obj["n"])
            leaf = Leaf(i=i, x=x, d=d, n=n, u=node)
            setattr(node, side, leaf)
            duplicates[leaf] = obj["ixs"]
        else:
            raise TypeError("`type` must be Branch or Leaf")

    @classmethod
    def from_dict(cls, obj):
        """
        Deserializes a nested dict representing an RCTree and creates a new
        RCTree instance from the loaded data.

        Parameters:
        -----------
        obj: dict
             Nested dictionary representing all nodes in the RCTree.

        Returns:
        --------
        newinstance: rrcf.RCTree
                     A new RCTree instance based on the loaded data.

        Example:
        --------
        # Load dict (see to_dict method for more info)
        >>> import json
        >>> with open('tree.json', 'r') as infile:
                obj = json.load(infile)

        # Create empty RCTree and load data
        >>> tree = rrcf.RCTree.from_dict(obj)

        # View loaded data
        >>> print(tree)
        >>>
        ─+
        ├───+
        │   ├──(3)
        │   └───+
        │       ├──(2)
        │       └──(0)
        └──(1)
        """
        newinstance = cls()
        newinstance.load_dict(obj)
        return newinstance

    def _lr_branch_bbox(self, node):
        """
        Compute bbox of node based on bboxes of node's children.
        """
        bbox = np.vstack(
            [
                np.minimum(node.l.b[0, :], node.r.b[0, :]),
                np.maximum(node.l.b[-1, :], node.r.b[-1, :]),
            ]
        )
        return bbox

    def _get_bbox_top_down(self, node):
        """
        Recursively compute bboxes of all branches from root to leaves.
        """
        if isinstance(node, Branch):
            if node.l:
                self._get_bbox_top_down(node.l)
            if node.r:
                self._get_bbox_top_down(node.r)
            bbox = self._lr_branch_bbox(node)
            node.b = bbox

    def _count_all_top_down(self, node):
        """
        Recursively compute number of leaves below each branch from
        root to leaves.
        """
        if isinstance(node, Branch):
            if node.l:
                self._count_all_top_down(node.l)
            if node.r:
                self._count_all_top_down(node.r)
            node.n = node.l.n + node.r.n

    def _count_leaves(self, node):
        """
        Count leaves underneath a single node.
        """
        num_leaves = np.array(0, dtype=np.int64)
        self.map_leaves(node, op=self._accumulate, accumulator=num_leaves)
        num_leaves = num_leaves.item()
        return num_leaves

    def _query(self, point, node):
        """
        Recursively search for the nearest leaf to a given point.
        """
        if isinstance(node, Leaf):
            return node
        else:
            if point[node.q] <= node.p:
                return self._query(point, node.l)
            else:
                return self._query(point, node.r)

    def _increment_depth(self, x, inc=1):
        """
        Primitive function for incrementing the depth attribute of a leaf.
        """
        x.d += inc

    def _accumulate(self, x, accumulator):
        """
        Primitive function for helping to count the number of points in a subtree.
        """
        accumulator += x.n

    def _get_nodes(self, x, stack):
        """
        Primitive function for listing all leaves in a subtree.
        """
        stack.append(x)

    def _get_bbox(self, x, mins, maxes):
        """
        Primitive function for computing the bbox of a point.
        """
        lt = x.x < mins
        gt = x.x > maxes
        mins[lt] = x.x[lt]
        maxes[gt] = x.x[gt]

    def _tighten_bbox_upwards(self, node):
        """
        Called when new point is inserted. Expands bbox of all nodes above new point
        if point is outside the existing bbox.
        """
        bbox = self._lr_branch_bbox(node)
        node.b = bbox
        node = node.u
        while node:
            lt = bbox[0, :] < node.b[0, :]
            gt = bbox[-1, :] > node.b[-1, :]
            lt_any = lt.any()
            gt_any = gt.any()
            if lt_any or gt_any:
                if lt_any:
                    node.b[0, :][lt] = bbox[0, :][lt]
                if gt_any:
                    node.b[-1, :][gt] = bbox[-1, :][gt]
            else:
                break
            node = node.u

    def _relax_bbox_upwards(self, node, point):
        """
        Called when point is deleted. Contracts bbox of all nodes above deleted point
        if the deleted point defined the boundary of the bbox.
        """
        while node:
            bbox = self._lr_branch_bbox(node)
            if not ((node.b[0, :] == point) | (node.b[-1, :] == point)).any():
                break
            node.b[0, :] = bbox[0, :]
            node.b[-1, :] = bbox[-1, :]
            node = node.u

    def _insert_point_cut(self, point, bbox):
        """
        Generates the cut dimension and cut value based on the InsertPoint algorithm.

        Parameters:
        -----------
        point: np.ndarray (1 x d)
               New point to be inserted.
        bbox: np.ndarray(2 x d)
              Bounding box of point set S.

        Returns:
        --------
        cut_dimension: int
                       Dimension to cut over.
        cut: float
             Value of cut.

        Example:
        --------
        # Generate cut dimension and cut value
        >>> _insert_point_cut(x_inital, bbox)

        (0, 0.9758881798109296)
        """
        # Generate the bounding box
        bbox_hat = np.empty((2, self.ndim), dtype=float)
        # Update the bounding box based on the internal point
        bbox_hat[0, :] = np.minimum(bbox[0, :], point)
        bbox_hat[-1, :] = np.maximum(bbox[-1, :], point)
        b_span = bbox_hat[-1, :] - bbox_hat[0, :]
        b_range = b_span.sum()
        r = self.rng.uniform(0, b_range)
        span_sum = np.cumsum(b_span)
        cut_dimension = np.inf
        for j in range(len(span_sum)):
            if span_sum[j] >= r:
                cut_dimension = j
                break
        if not np.isfinite(cut_dimension):
            raise ValueError("Cut dimension is not finite.")
        cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
        return cut_dimension, cut


class Branch:
    """
    Branch of RCTree containing two children and at most one parent.

    Attributes:
    -----------
    q: Dimension of cut
    p: Value of cut
    l: Pointer to left child
    r: Pointer to right child
    u: Pointer to parent
    n: Number of leaves under branch
    b: Bounding box of points under branch (2 x d)
    """

    __slots__ = ["q", "p", "l", "r", "u", "n", "b"]

    def __init__(self, q, p, l=None, r=None, u=None, n=0, b=None):
        self.l = l
        self.r = r
        self.u = u
        self.q = q
        self.p = p
        self.n = n
        self.b = b

    def __repr__(self):
        return "Branch(q={}, p={:.2f})".format(self.q, self.p)


class Leaf:
    """
    Leaf of RCTree containing no children and at most one parent.

    Attributes:
    -----------
    i: Index of leaf (user-specified)
    d: Depth of leaf
    u: Pointer to parent
    x: Original point (1 x d)
    n: Number of points in leaf (1 if no duplicates)
    b: Bounding box of point (1 x d)
    """

    __slots__ = ["i", "d", "u", "x", "n", "b"]

    def __init__(self, i, d=None, u=None, x=None, n=1):
        self.u = u
        self.i = i
        self.d = d
        self.x = x
        self.n = n
        self.b = x.reshape(1, -1)

    def __repr__(self):
        return "Leaf({0})".format(self.i)
