# vim: expandtab:ts=4:sw=4
"""
This module contains helper functions to write dataset structures to a simple
CSV file format that is compatible with the MOT challenge SDK [1]_.

The MOT challenge is a multiple object tracking benchmark that aims to establish
standardized multiple object tracking evaluation on wide ringe of datasets.

.. [1] https://motchallenge.net/
"""
import os
import numpy as np

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
        of detections in that frame. If the detection file contains 3D
        positions, these will be used as sensor_data. Otherwise, the sensor_data
        is set to the detection's region of interest (ROI).

    """
    # format: frame id, track id, bbox (x, y, w, h), confidence, world (x, y, z)
    # track id is always -1
    data = np.loadtxt(filename, delimiter=',')
    has_threed = np.any(data[:, 7:10] != -1)
    min_frame_idx = int(data[:, 0].min())
    max_frame_idx = int(data[:, 0].max())
    detections = {i: [] for i in range(min_frame_idx, max_frame_idx + 1)}
    for row in data:
        confidence = row[6]
        if min_confidence is not None and row[6] < min_confidence:
            continue
        frame_idx, roi = int(row[0]), row[2:6]
        xyz = row[7:10] if has_threed else None
        detections[frame_idx].append(
            pymotutils.RegionOfInterestDetection(
                frame_idx, roi, confidence, xyz=xyz))
    return detections


def read_groundtruth(filename, sensor_data_is_3d=False):
    """Read ground truth file.

    Parameters
    ----------
    filename : str
        Path to the ground truth file.
    sensor_data_is_3d : bool
        If True, the ground truth's sensor data is set to the 3D position.
        If False, the ground truth's sensor data is set to the region of
        interest (ROI).

        Note that not all of the sequences provided by the MOT challenge contain
        valid 3D positions.

    Returns
    -------
    TrackSet
        Returns the tracking ground truth. If sensor_data_is_3d is True, the
        sensor data contains the 3D position. Otherwise, sensor_data
        is set to the region of interest (ROI).

    """
    # format: frame id, track id, bbox (x, y, w, h), care_flag, world (x, y, z)
    if not os.path.isfile(filename):
        return pymotutils.TrackSet()
    data = np.loadtxt(filename, delimiter=',')

    has_threed = np.any(data[:, 7:10] != -1)
    if sensor_data_is_3d and not has_threed:
        raise RuntimeError("File does not contain valid 3D coordinates")

    ground_truth = pymotutils.TrackSet()
    for row in data:
        frame_idx, track_id = int(row[0]), int(row[1])
        do_not_care = row[6] == 0
        if sensor_data_is_3d:
            sensor_data = row[7:10]
        else:
            sensor_data = row[2:6]
        if track_id not in ground_truth.tracks:
            ground_truth.create_track(track_id)
        ground_truth.tracks[track_id].add(
            pymotutils.Detection(frame_idx, sensor_data, do_not_care))
    return ground_truth


def write_hypotheses(filename, track_set_2d=None, track_set_3d=None):
    """Write track hypotheses (tracking output) to file.

    This function supports writing of track hypotheses files compatible with
    2D or 3D evaluation, or both. In the 2D case, the track set should contain
    the image region of interest (ROI). In the 3D case, the track set should
    contain the 3D position in the tracking frame.

    Note that the MOT challenge devkit requires that sequences startat index 1.
    This function will automatically correct the index accordingly.

    Parameters
    ----------
    filename : str
        Name of the file to write to (file format will be CSV).
    track_set_2d : Optional[TrackSet]
        The set of track hypotheses (tracking output), where sensor_data
        contains the object's region of interest (ROI).
    track_set_3d : Optional[TrackSet]
        The set of track hypotheses (tracking output), where sensor_data
        contains the object's 3D position.

    """
    ref_set = track_set_2d if track_set_2d is not None else track_set_3d
    offset = 1 - ref_set.first_frame_idx()

    csvfile = open(filename, "w")
    for frame_idx in ref_set.frame_range():
        if track_set_2d is not None:
            data_2d = track_set_2d.collect_sensor_data(frame_idx)
        else:
            data_2d = {}
        if track_set_3d is not None:
            data_3d = track_set_3d.collect_sensor_data(frame_idx)
        else:
            data_3d = {}
        track_ids = set(data_2d.keys()) | set(data_3d.keys())
        for track_id in track_ids:
            if track_id in data_2d:
                bbox = list(data_2d[track_id])
            else:
                bbox = [-1, -1, -1, -1]
            if track_id in data_3d:
                world = list(data_3d[track_id])
            else:
                world = [-1, -1, -1]
            row = [frame_idx + offset, track_id] + bbox + [-1] + world
            csvfile.writelines(",".join(str(x) for x in row) + os.linesep)
    csvfile.close()


def write_groundtruth(filename, track_set_2d=None, track_set_3d=None):
    """Write ground truth data to file.

    This function supports writing of ground truth files compatible with
    2D or 3D evaluation, or both. In the 2D case, the track set should contain
    the image region of interest (ROI). In the 3D case, the track set should
    contain the 3D position in the tracking frame.

    Note that the MOT challenge devkit requires that sequences startat index 1.
    This function will automatically correct the index accordingly.

    Parameters
    ----------
    filename : str
        Name of the file to write to (file format will be CSV)
    track_set_2d : Optional[TrackSet]
        The set of ground truth tracks, where sensor_data contains the
        image region of interest (ROI).
    track_set_3d : Optional[TrackSet]
        The set of ground truth tracks, where sensor_data contains the
        objects' 3D position.

    """
    ref_set = track_set_2d if track_set_2d is not None else track_set_3d
    offset = 1 - ref_set.first_frame_idx()

    csvfile = open(filename, "w")
    for frame_idx in ref_set.frame_range():
        if track_set_2d is not None:
            data_2d = track_set_2d.collect_detections(frame_idx)
        else:
            data_2d = {}
        if track_set_3d is not None:
            data_3d = track_set_3d.collect_detections(frame_idx)
        else:
            data_3d = {}
        track_ids = set(data_2d.keys()) | set(data_3d.keys())
        for track_id in track_ids:
            care_flag = True
            if track_id in data_2d:
                bbox = list(data_2d[track_id].sensor_data)
                care_flag = care_flag and not data_2d[track_id].do_not_care
            else:
                bbox = [-1, -1, -1, -1]
            if track_id in data_3d:
                world = list(data_3d[track_id].sensor_data)
                care_flag = care_flag and not data_3d[track_id].do_not_care
            else:
                world = [-1, -1, -1]

            care_int = 1 if care_flag else 0
            row = [frame_idx + offset, track_id] + bbox + [care_int] + world
            csvfile.writelines(",".join(str(x) for x in row) + os.linesep)
    csvfile.close()
