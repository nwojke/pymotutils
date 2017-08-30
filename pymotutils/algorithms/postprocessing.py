# vim: expandtab:ts=4:sw=4
"""
This module contains functionality to postprocess the tracking output.
"""
import numpy as np
import pymotutils


def convert_track_set(track_set, detection_converter):
    """Create a track set copy with modified detections.

    Using this function, you can modify the detections contained in a track set
    using a user-specified converter function.

    For example, you may wish to set the sensor_data attribute to an application
    specific field 'roi' in order to interpolate this data:

    >>> track_set = create_my_tracking_hypotheses()
    >>> dataset_converter = lambda d: Detection(d.frame_idx, d.roi)
    >>> roi_data = convert_track_set(track_set, dataset_converter)
    >>> interpolated = interpolate_track_set(roi_data)

    Parameters
    ----------
    track_set : TrackSet
        The input track set.
    detection_converter : Callable[Detection] -> Detection
        The converter function. This is called once for each detection contained
        in the track set.

    Returns
    -------
    TrackSet
        Returns the converted track set with the same structure as the input
        track set, but where each detection has been converted.

    """
    result = pymotutils.TrackSet()
    for tag, track in track_set.tracks.items():
        result_track = result.create_track(tag)
        for detection in track.detections.values():
            result_track.add(detection_converter(detection))
    return result


def interpolate_track_set(track_set):
    """Interpolate sensor data in given track set.

    This method uses linear interpolation to fill missing detections in each
    track of the given track set. Each dimension of the sensor data is
    interpolated independently of all others.

    For example, if the sensor data contains 3-D positions, then the X, Y, and Z
    coordinates of the trajectory are interpolated linearily. The same method
    works fairly well for image regions as well.

    Parameters
    ----------
    track_set : TrackSet
        The track set to be interpolated. The sensor data must be an array_like
        (ndim=1).

    Returns
    -------
    TrackSet
        The interpolated track set, where each target is visible from the first
        frame of appearance until leaving the scene.

    """
    interp_set = pymotutils.TrackSet()
    for tag, track in track_set.tracks.items():
        first, last = track.first_frame_idx(), track.last_frame_idx()
        frame_range = np.arange(first, last + 1)
        xp = sorted(list(track.detections.keys()))

        if len(xp) == 0:
            continue  # This is an empty trajectory.

        sensor_data = np.asarray([track.detections[i].sensor_data for i in xp])
        fps = [sensor_data[:, i] for i in range(sensor_data.shape[1])]
        interps = [np.interp(frame_range, xp, fp) for fp in fps]

        itrack = interp_set.create_track(tag)
        for i, frame_idx in enumerate(frame_range):
            sensor_data = np.array([interp[i] for interp in interps])
            do_not_care = frame_idx not in track.detections
            itrack.add(pymotutils.Detection(frame_idx, sensor_data, do_not_care))
    return interp_set
