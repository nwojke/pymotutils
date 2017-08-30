# vim: expandtab:ts=4:sw=4
"""
This module contains code for solving the linear assingment problem (bipartite
matching) as well as some useful cost functions.
"""
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as la_solver

_INFTY_COST = 1e+5
_EPS_COST = 1e-5


def _intersection_over_union(roi, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    roi : ndarray
        A bounding box in format (top left x, top left y, width, height).
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as roi.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the given roi and each
        candidate. A higher score means a larger fraction of the roi is
        occluded by the candidate.

    """
    roi_tl, roi_br = roi[:2], roi[:2] + roi[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(roi_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(roi_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(roi_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(roi_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_roi = roi[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_roi + area_candidates - area_intersection)


def intersection_over_union_cost(rois_a, rois_b):
    """Computer intersection over union.

    Parameters
    ----------
    rois_a: ndarray
        An Nx4 dimensional array of bounding boxes in format (top left x,
        top left y, width, height).
        An Mx4 dimensional array of bounding boxes in format (top left x,
        top left y, width, height).

    Returns
    -------
    ndarray
        A cost matrix of shape NxM where element (i, j) contains `1 - iou`
        between the i-th roi in rois_a and the j-th roi in rois_b (a larger
        score means less bounding box overlap).

    """
    cost_matrix = np.zeros((len(rois_a), len(rois_b)))
    for i, roi in enumerate(rois_a):
        cost_matrix[i, :] = 1.0 - _intersection_over_union(roi, rois_b)
    return cost_matrix


def min_cost_matching(cost_matrix, max_cost=None):
    """Solve a linear assignment problem.

    Parameters
    ----------
    cost_matrix : ndarray
        An NxM matrix where element (i,j) contains the cost of matching
        the i-th element out of the first set of N elements to the j-th element
        out of the second set of M elements.
    max_cost: float
        Gating threshold. Associations with cost larger than this value are
        disregarded.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Returns a tuple with the following three entries:
        * An array of shape Lx2 of matched elements (row index, column index).
        * An array of unmatched row indices.
        * An array of unmatched column indices.

    """
    if max_cost is not None:
        cost_matrix[cost_matrix > max_cost] = max_cost + _EPS_COST
    matched_indices = la_solver(cost_matrix)
    if max_cost is not None:
        row_indices, col_indices = matched_indices[:, 0], matched_indices[:, 1]
        mask = cost_matrix[row_indices, col_indices] <= max_cost
        matched_indices = matched_indices[mask, :]

    # TODO(nwojke): I think there is a numpy function for the set difference
    # that is computed here (it might also be possible to speed this up if
    # sklearn preserves the order of indices, which it does?).
    unmatched_a = np.array(
        list(set(range((cost_matrix.shape[0]))) - set(matched_indices[:, 0])))
    unmatched_b = np.array(
        list(set(range((cost_matrix.shape[1]))) - set(matched_indices[:, 1])))

    return matched_indices, unmatched_a, unmatched_b
