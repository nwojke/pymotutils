# vim: expandtab:ts=4:sw=4
"""
This module contains functionality to preprocess detections.
"""
import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick


def filter_detections(
        detection_dict, min_confidence=None, min_width=None, min_height=None):
    """Filter detections by detector confidencen and extent.

    Parameters
    ----------
    detection_dict : Dict[int, List[Detection]]
        A dictionary that maps from frame index to a list of detections.
    min_confidence : Optional[float]
        If not None, discards detections with confidence lower than this
        value.
    min_width : Optional[float]
        If not None, discards detections with width lower than this value.
    min_height : Optional[float]
        If not None, discards detections with height lower than this value.

    Returns
    -------
    Dict[int, List[Detection]]
        Returns the dictionary of filtered detections.

    """
    if min_width is None:
        min_width = -np.inf
    if min_height is None:
        min_height = -np.inf

    def filter_fn(detection):
        if min_confidence is not None and detection.confidence < min_confidence:
            return False
        if detection.roi[2] < min_width:
            return False
        if detection.roi[3] < min_height:
            return False
        return True

    filtered_detection_dict = {}
    for frame_idx in detection_dict.keys():
        filtered_detection_dict[frame_idx] = list(
            filter(filter_fn, detection_dict[frame_idx]))
    return filtered_detection_dict
