# vim: expandtab:ts=4:sw=4
"""
This module contains helper functions to write dataset structures to a JSON
file format that is compatible with pymot [1]_.

Pymot is an open source tool for evaluation of multiple object tracking
performance using CLEAR MOT metrics.

.. [1] https://github.com/Videmo/pymot
"""
import json
import pymotutils


def write_groundtruth(filename, track_set):
    """Write ground truth data to file.

    It is assumed that the sensor data contained in each of the tracks
    is a region of interest (x, y, width, height).

    Parameters
    ----------
    filename : str
        Name of the file to write to (file format will be JSON).
    track_set : dataset.TrackSet
        The set of ground truth tracks.

    """
    assert isinstance(track_set, pymotutils.TrackSet), (
        "track_set is of wrong type")

    output = {"frames": []}
    for frame_id in track_set.frame_range():
        frame = dict()
        frame["timestamp"] = frame_id
        frame["class"] = "frame"
        frame["annotations"] = []

        detections = track_set.collect_detections(frame_id)
        for tag, detection in detections.items():
            annotation = dict()
            annotation["dco"] = bool(detection.do_not_care)
            annotation["x"] = float(detection.sensor_data[0])
            annotation["y"] = float(detection.sensor_data[1])
            annotation["width"] = float(detection.sensor_data[2])
            annotation["height"] = float(detection.sensor_data[3])
            annotation["id"] = tag
            frame["annotations"].append(annotation)
        output["frames"].append(frame)
    output["class"] = "video"

    with open(filename, "w") as f:
        json.dump([output], f, indent=4, sort_keys=True)


def write_hypotheses(filename, track_set):
    """Write track hypotheses (tracking output) to file.

    It is assumed that the sensor data contained in each of the tracks
    is a region of interest (x, y, width, height).

    Parameters
    ----------
    filename : str
        Name of the file to write to (file format will be JSON).
    track_set : dataset.TrackSet
        The set of track hypotheses (tracking output).

    """
    assert isinstance(track_set, dataset.TrackSet), (
        "track_set is of wrong type")

    output = {"frames": []}
    for frame_id in track_set.frame_range():
        frame = dict()
        frame["timestamp"] = frame_id
        frame["class"] = "frame"
        frame["hypotheses"] = []

        detections = track_set.collect_detections(frame_id)
        for tag, detection in detections.items():
            hypothesis = dict()
            hypothesis["x"] = float(detection.sensor_data[0])
            hypothesis["y"] = float(detection.sensor_data[1])
            hypothesis["width"] = float(detection.sensor_data[2])
            hypothesis["height"] = float(detection.sensor_data[3])
            hypothesis["id"] = tag
            frame["hypotheses"].append(hypothesis)
        output["frames"].append(frame)
    output["class"] = "video"

    with open(filename, "w") as f:
        json.dump([output], f, indent=4, sort_keys=True)
