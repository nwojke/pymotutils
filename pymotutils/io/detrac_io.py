# vim: expandtab:ts=4:sw=4
"""
This module contains helper functions to read and write dataset structures in a
way that is compatible with the DETRAC-toolkit [1]_.

.. [1] http://detrac-db.rit.albany.edu/
"""
import numpy as np
import xml.etree.ElementTree as ElementTree
import os
from six import itervalues

import pymotutils


def read_detections(filename, min_confidence=None):
    """Read detection file.

    Parameters
    ----------
    filename : str
        Path to the detection file.
    min_confidence : Optional[float]
        A detector confidence threshold. Detections with lower confidence are
        disregarded.

    Returns
    -------
    Dict[int, List[MonoDetection]]
        This function returns a dictionary that maps frame indices to a list
        of detections in that frame.

    """
    # format: frame id, bbox (x, y, w, h), confidence
    data = np.loadtxt(filename, delimiter=',')
    min_frame_idx = int(data[:, 0].min())
    max_frame_idx = int(data[:, 0].max())
    detections = {i: [] for i in range(min_frame_idx, max_frame_idx + 1)}
    for row in data:
        confidence = row[6]
        if min_confidence is not None and confidence < min_confidence:
            continue
        frame_idx, roi = int(row[0]), row[2:6]
        detections[frame_idx].append(
            pymotutils.RegionOfInterestDetection(
                frame_idx, roi, confidence))
    return detections


def read_groundtruth(filename):
    """Read ground truth data.

    Parameters
    ----------
    filename : str
        Path to the ground truth file.

    Returns
    -------
    dataset.TrackSet
        Returns the tracking ground truth. However the ground truth file
        contains other useful information, the sensor_data contains only the ROI
        [left, top, width, height].

    """
    if not os.path.isfile(filename):
        return pymotutils.TrackSet()
    tree = ElementTree.parse(filename)

    ground_truth = pymotutils.TrackSet()
    sequence = tree.getroot()
    for frame in sequence.iter('frame'):
        for target in frame.iter('target'):
            frame_idx = int(frame.get('num'))
            track_id = int(target.get('id'))
            box = target.find('box')
            sensor_data = np.asarray(
                [float(box.get('left')),
                 float(box.get('top')),
                 float(box.get('width')),
                 float(box.get('height'))])

            if track_id not in ground_truth.tracks:
                ground_truth.create_track(track_id)
            ground_truth.tracks[track_id].add(
                pymotutils.Detection(frame_idx, sensor_data))
    return ground_truth


def write_hypotheses(foldername, sequence_name, track_set, speed=25.0):
    """Write track hypotheses (tracking output) to files.

     The DETRAC toolkit expect tracking result in 5 separate files per sequence.
     More info here: [1]_.

     .. [1] http://detrac-db.rit.albany.edu/instructions

    Parameters
    ----------
    foldername : str
        Path to the folder to store CSVs.
    sequence_name : str
        Name of the current sequence. The DETRAC toolkit expects all results in
        a single folder, and uses this name as a prefix to separate sequences.
    track_set : dataset.TrackSet
        The set of track hypotheses (tracking output), where sensor_data
        contains the object's region of interest (ROI).
    speed : Optional[float]
        Running speed of the tracker in frame per sec (FPS). If not specified, a
        a dummy value will be used.

    Returns
    -------

    """
    num_of_frames = track_set.last_frame_idx()
    track_array = np.zeros((4, num_of_frames, len(track_set.tracks)))
    for (track_idx, track_id) in enumerate(track_set.tracks):
        for obj in itervalues(track_set.tracks[track_id].detections):
            track_array[:, obj.frame_idx-1, track_idx] = obj.sensor_data[0:4]

    for i, suffix in enumerate(['_LX.txt', '_LY.txt', '_W.txt', '_H.txt']):
        np.savetxt(
            os.path.join(foldername, sequence_name + suffix),
            track_array[i],
            fmt='%.3g', delimiter=',')

    np.savetxt(os.path.join(foldername, sequence_name + '_Speed.txt'), [speed],
               fmt='%.5f')


def write_groundtruth():
    raise NotImplementedError()
