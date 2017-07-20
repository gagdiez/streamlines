import builtins
import collections

import numpy as np
import scipy.interpolate

from .asarray import distance, hash, length, reorient, resample


class Streamline(object):
    """A diffusion MRI streamline"""

    def __init__(self, points=None):

        if points is None:
            points = np.empty((0, 3))
        else:
            try:
                points = np.array(points, dtype=float)
            except:
                raise TypeError(
                    'points must be convertible to a numpy array of floats.')

        if points.ndim != 2:
            raise ValueError(
                'points must be a two dimensionnal array, not {} dimensionnal.'
                .format(points.ndim))

        if points.shape[1] != 3:
            raise ValueError(
                'points must have a shape of (N, 3), not {}.'
                .format(points.shape))

        self._points = points

    def __contains__(self, point):
        """Verifies if a point is part of a streamline"""
        return next((True for p in self._points if np.all(p == point)), False)

    def __eq__(self, other):
        return hash(self._points) == hash(other._points)

    def __hash__(self):
        return hash(self._points)

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def __reversed__(self):
        return reversed(self._points)

    def __str__(self):
        return 'streamline: {} points'.format(len(self))

    @property
    def points(self):
        return self._points.copy()

    @property
    def length(self):
        return length(self._points)

    def distance(left, right, nb_points=20):
        return distance(left._points, right._points, nb_points)

    def reorient(self, template):
        return Streamline(reorient(self._points, template._points))

    def resample(self, nb_points):
        return Streamline(resample(self._points, nb_points))


class Streamlines(object):
    """A sequence of dMRI streamlines"""

    def __init__(self, iterable=None, affine=np.eye(4)):
        
        self.affine = affine
        self._items = []
        if iterable is not None:
            self._items = [Streamline(i) for i in iterable]

    def __iadd__(self, other):
        self._items += other._items
        return self

    def __contains__(self, streamline):
        return streamline in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __str__(self):
        return str(self._items)

    def filter(self, min_length=None):

        if min_length is not None:
            self._items = [i for i in self._items if i.length >= min_length]