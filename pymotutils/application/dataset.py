# vim: expandtab:ts=4:sw=4
"""
The data structures in this module may be used to store detections, ground
truth data, and tracking hypotheses for a particular dataset.
"""
import numpy as np
import pymotutils


class Detection(object):
    """This is a container class for object detections and ground truth data.

    Parameters
    ----------
    frame_idx : int
        Index of the frame at which the detection occured
    sensor_data : array_like
        Sensor data (application dependent). Could be, e.g., a point
        measurement or a region of interest in an image. The sensor data
        must be representable as vector of floats. If you have other
        application data (e.g., for visualization), you can pass them
        through kwargs.
    do_not_care : bool
        This flag indicates whether this detection should be included
        in evaluation.
        If True, missing this detection will be counted as false negative. If
        False, missing this detection will not have a negative impact on
        tracking performance. Therefore, this flag can be used to mark hard to
        detect objects (such as full occlusions) in ground truth.
    kwargs : Optional[Dict[str, T]]
        Optional keyname arguments that will be accissible as attributes of
        this class.

    Attributes
    ----------
    frame_idx : int
        Index of the frame at which the detection occured.
    sensor_data : ndarray
        Sensor data (application dependent). Could be, e.g., a point
        measurement or a region of interest in an image.
    do_not_care : bool
        This flag indicates whether this detection is should be included
        in evaluation.
        If True, missing this detection will be counted as false negative.
        If False, missing this detection will not have an impact on tracking
        performance.
        Therefore, this flag can be used to mark hard to detect objects (such
        as full occlusions) in ground truth.

    """

    def __init__(self, frame_idx, sensor_data, do_not_care=False, **kwargs):
        self.frame_idx = frame_idx
        self.sensor_data = np.asarray(sensor_data)
        self.do_not_care = do_not_care
        for key, attr in kwargs.items():
            setattr(self, key, attr)


class Track(object):
    """A track is an object identity that appears within the dataset.

    Parameters
    ----------
    detections : Optional[Dict[int, Detection]]
        A dictionary of detections. The key is the frame index at which the
        detection occured. The value is the detection object itself.
        If None, an empty dictionary is created.

    Attributes
    ----------
    detections : Dict[int, Detection]
        A dictionary of detections. The key is the frame index at which the
        detection occured, the value is the detection object itself.
        You may directly modify this attribute.

    """

    def __init__(self, detections=None):
        if detections is None:
            detections = {}
        self.detections = detections

    def add(self, detection):
        """Add a detection to the track.

        Parameters
        ----------
        detection : Detection
            The detection to add.

        """
        assert isinstance(detection, Detection), "Detection is of wrong type"
        assert detection.frame_idx not in self.detections, "duplicate frame_idx"
        self.detections[detection.frame_idx] = detection

    def first_frame_idx(self):
        """Get index of the first frame at which the object appears.

        Returns
        -------
        int
            Index of the first frame at which the object appears.

        """
        return 0 if len(self.detections) == 0 else min(self.detections.keys())

    def last_frame_idx(self):
        """Get index of the last frame at which the object is present.

        Returns
        -------
        int
            Index of the last frame at which the object is present.

        """
        return 0 if len(self.detections) == 0 else max(self.detections.keys())

    def num_frames(self):
        """Get the total number of frames this object appears in, including
         occlusions.

         Returns
         -------
         int
            self.last_frame_idx() - self.first_frame_idx() + 1

        """
        return self.last_frame_idx() - self.first_frame_idx() + 1

    def frame_range(self):
        """ Get range from first frame of appearance to last frame of presence.

        Returns
        -------
        range
            >>> range(self.first_frame_idx(), self.last_frame_idx() + 1)

        """
        start, end = self.first_frame_idx(), self.last_frame_idx() + 1
        return range(start, end)

    def is_in_frame(self, frame_idx):
        """Test whether the object has been detected in a given frame.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to test against.

        Returns
        -------
        bool
            True if the object is present at the given frame.

        """
        return frame_idx in self.detections


class TrackSet(object):
    """
    A set of multiple tracks. Each track is identified by a unique index (tag).

    Parameters
    ----------
    tracks : Optional[Dict[int, Track]]
        Mapping from track identifier (also called track id or tag) to Track
        object. If None, an empty dictionary is created.

    Attributes
    ----------
    tracks : Dict[int, Track]
        Mapping from track id (also called tag) to Track object. You may
        directly manipulate this attribute.

    """

    def __init__(self, tracks=None):
        self.tracks = tracks if tracks is not None else {}

    def create_track(self, tag):
        """Create a new track.

        The newly created track is added to the track set.

        Parameters
        ----------
        tag : int
            A unique object identifier. None of the existing tracks must share
            the same tag.

        Returns
        -------
        Track
            The newly created track.

        """
        assert tag not in self.tracks, "track with tag %d exists already" % tag
        self.tracks[tag] = Track()
        return self.tracks[tag]

    def first_frame_idx(self):
        """Get the index of the first frame at which any object is present.

        Returns
        -------
        int
            Index of the first frame at which any object is present.

        """
        return 0 if len(self.tracks) == 0 else min(
            track.first_frame_idx() for track in self.tracks.values())

    def last_frame_idx(self):
        """Get the index of the last frame at which any object is present.

        Returns
        -------
        int
            Index of the last frame at which any object is present.

        """
        return 0 if len(self.tracks) == 0 else max(
            track.last_frame_idx() for track in self.tracks.values())

    def num_frames(self):
        """Get the total number of frames at which at least one object is
        present, including occlusions.

        Returns
        -------
        int
            self.last_frame_idx() - self.first_frame_idx() + 1

        """
        return self.last_frame_idx() - self.first_frame_idx() + 1

    def frame_range(self):
        """ Get range of frames where at least one object is present.

        Returns
        -------
        range
            Range from first frame at which any object appears to last frame at
            which any object is present::

            >>> range(self.first_frame_idx(), self.last_frame_idx() + 1)

        """
        start, end = self.first_frame_idx(), self.last_frame_idx() + 1
        return range(start, end)

    def collect_detections(self, frame_idx):
        """Collect all detections for a given frame index.

        Parameters
        ----------
        frame_idx : int
            Index of the frame for which to collect detections.

        Returns
        -------
        Dict[int, Detection]
            A mapping from track identifier (tag) to Detection.
        """
        detections = {}
        for tag, track in self.tracks.items():
            if frame_idx not in track.detections:
                continue
            detections[tag] = track.detections[frame_idx]
        return detections

    def collect_sensor_data(self, frame_idx):
        """Collect all sensor data for a given frame index.

        Parameters
        ----------
        frame_idx : int
            Index of the frame for which to collect sensor data.

        Returns
        -------
        Dict[int, T]
            A mapping from track identifier (tag) to sensor data.
        """
        sensor_data = {}
        for tag, track in self.tracks.items():
            if frame_idx not in track.detections:
                continue
            sensor_data[tag] = track.detections[frame_idx].sensor_data
        return sensor_data


def iterate_track_pairwise_with_time_offset(track, time_offset, for_each):
    """Generate all pairs of detections contained in a track.

    This function may be used to obtain positive examples for training the
    parameters of a pairwise matching cost function.

    Parameters
    ----------
    track : Track
        A track to iterate over.
    time_offset : int
        The specified time offset between all pairs of detections.
    for_each : Callable[Detection, Detection] -> None
        A function that will be called for each pair. The first argument is
        the detection with smaller frame index, the second argument is the
        matching detection according to the time_offset.

    """
    for frame_idx in track.frame_range():
        ahead_idx = frame_idx + time_offset
        if frame_idx not in track.detections:
            continue
        if ahead_idx not in track.detections:
            continue
        for_each(track.detections[frame_idx], track.detections[ahead_idx])


def iterate_track_set_with_time_offset(track_set, time_offset, for_each):
    """Generate all pairs of detections contained in a track set.

    This function may be used to obtain positive and negative examples for
    training the parameters of a pairwise matching cost function.

    Parameters
    ----------
    track_set : TrackSet
        The track set to iterate over.
    time_offset : int
        The specified time offset between all pairs of detections.
    for_each : Callable[int, Detection, int, Detection] -> None
        A function that will be called for each pair. The first two arguments
        are the track id and detection with smaller frame index, the third
        and fourth argument are a matching track id and detection.

    """
    for frame_idx in track_set.frame_range():
        ahead_idx = frame_idx + time_offset

        detections_now = track_set.collect_detections(frame_idx)
        detections_ahead = track_set.collect_detections(ahead_idx)

        for track_id_i, detection_i in detections_now.items():
            for track_id_k, detection_k in detections_ahead.items():
                for_each(track_id_i, detection_i, track_id_k, detection_k)


def associate_detections(ground_truth, detections, min_bbox_overlap=0.5):
    """Associate detections to ground truth tracks.

    Parameters
    ----------
    ground_truth : TrackSet
        The ground truth track set. Each detection must contain a region
        of interest in format (top left x, top left y, width, height)` in
        the sensor_data attribute.
    detections : Dict[int, List[Detection]]
        A dictionary that maps from frame index to list of detections. Each
        detection must contain an attribute `roi` that contains the region
        of interest in the same format as the ground_truth.
    min_bbox_overlap : float
        The minimum bounding box overlap for valid associations. A larger value
        increases the misalignment between detections and ground truth.

    Returns
    -------
    (TrackSet, Dict[int, List[Detection]])
        The first element in the tuple is the set of detections associated with
        each ground truth track. The second element is a dictionary that maps
        from frame index to list of false alarms.

    """
    track_set = TrackSet()
    false_alarms = {}

    for frame_idx in ground_truth.frame_range():
        ground_truth_id_to_roi = ground_truth.collect_sensor_data(frame_idx)
        ground_truth_ids = list(ground_truth_id_to_roi.keys())
        ground_truth_rois = np.asarray([
            ground_truth_id_to_roi[k] for k in ground_truth_ids])
        detection_rois = np.asarray([
            d.roi for d in detections.get(frame_idx, [])])

        if ground_truth_rois.shape[0] == 0:
            ground_truth_rois = ground_truth_rois.reshape(0, 4)
        if detection_rois.shape[0] == 0:
            detection_rois = detection_rois.reshape(0, 4)

        cost_matrix = pymotutils.linear_assignment.intersection_over_union_cost(
            ground_truth_rois, detection_rois)
        matched_indices, _, unmatched_detections = (
            pymotutils.linear_assignment.min_cost_matching(
                cost_matrix, max_cost=1.0 - min_bbox_overlap))

        false_alarms[frame_idx] = [
            detections[frame_idx][i] for i in unmatched_detections]
        for ground_truth_idx, detection_idx in matched_indices:
            track_id = ground_truth_ids[ground_truth_idx]
            if track_id not in track_set.tracks:
                track_set.create_track(track_id)
            track_set.tracks[track_id].add(
                detections[frame_idx][detection_idx])

    return track_set, false_alarms
