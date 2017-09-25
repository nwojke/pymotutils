# vim: expandtab:ts=4:sw=4
"""
This module contains code for tracking applications using a single
camera.
"""
import numpy as np
import cv2
import pymotutils


class RegionOfInterestDetection(pymotutils.Detection):
    """
    A single-camera detection that contains the region of interest (ROI) and,
    optionally, a detector confidence score.

    Parameters
    ----------
    frame_idx : int
        Index of the frame at which this detection occured.
    roi : ndarray
        The region of interest in which the object is contained as 4
        dimensional vector (x, y, w, h) where (x, y) is the top-left corner
        and (w, h) is the extent.
    confidence : NoneType | float
        Optional detector confidence score. If not None, it is appended to the
        sensor_data field of the detection.
    xyz : NoneType | ndarray
        Optional object locataion, e.g., in camera or world frame.
    feature : NoneType | ndarray
        Optional appearance descriptor.
    do_not_care : bool
        This flag indicates whether this detection should be included
        in evaluation.
        If True, missing this detection will be counted as false negative. If
        False, missing this detection will not have a negative impact on
        tracking performance. Therefore, this flag can be used to mark hard to
        detect objects (such as full occlusions) in ground truth.

    Attributes
    ----------
    roi : ndarray
        The region of interest in which the object is contained as 4
        dimensional vector (x, y, w, h) where (x, y) is the top-left corner
        and (w, h) is the extent.
    confidence : NoneType | float
        Optinal detector confidence score
    xyz : NoneType | ndarray
        Optional object location, e.g., in camera or world frame.
    feature : NoneType | ndarray
        Optional appearance descriptor.

    """

    def __init__(
            self, frame_idx, roi, confidence=None, xyz=None, feature=None,
            do_not_care=False):
        sensor_data = roi if confidence is None else np.r_[roi, confidence]
        super(RegionOfInterestDetection, self).__init__(
            frame_idx, sensor_data, do_not_care=do_not_care, roi=roi,
            confidence=confidence, xyz=xyz, feature=feature)


class MonoVisualization(pymotutils.ImageVisualization):
    """
    This class implements an image-based visualization of tracking output
    obtained from a single camera.

    Parameters
    ----------
    update_ms : int
        Number of milliseconds to wait before processing the next time step.
    window_shape : Tuple[int, int]
        Shape of the image viewer in format (width, height).
    online_tracking_visualization : Optional[Callable[MonoVisualization, Dict[str, T], Tracker]]
        If not None, this function is called once at the end of each time step
        to visualize tracking output.

        The first argument is the visualization object (self), the second
        argument is frame_data dictionary of the current time step, and the
        third argument is the tracker that is processing the data.
    caption : str
        The window caption.

    Attributes
    ----------
    detection_thickness : int
        The line thickness to be used for drawing detections.
    detection_color : Tuple[int, int, int]
        The color to be used for drawing detections.
    track_set_thickness : int
        The line thickness to be used for drawing track sets (e.g.,
        ground truth or tracking output).
    draw_trajectories : bool
        If True, draws the entire trajectory of each object. If False, draws
        only the location at the current time step.

    """

    def __init__(
            self, update_ms, window_shape, online_tracking_visualization=None,
            caption="Figure 1"):
        super(MonoVisualization, self).__init__(
            update_ms, window_shape, caption)
        self.detection_thickness = 2
        self.detection_color = 0, 0, 255
        self.track_set_thickness = 2
        self.draw_trajectories = False
        self._frame_data = None

        if online_tracking_visualization is None:
            self._draw_online_tracking_results = (
                lambda image_viewer, frame_data, tracker: None)
        else:
            self._draw_online_tracking_results = online_tracking_visualization

    def init_frame(self, frame_data):
        self._viewer.image = frame_data["bgr_image"].copy()
        self._frame_data = frame_data

    def finalize_frame(self):
        self._frame_data = None

    def draw_detections(self, detections):
        self._viewer.thickness = self.detection_thickness
        self._viewer.color = self.detection_color
        for i, detection in enumerate(detections):
            x, y, w, h = detection.roi
            confidence = detection.confidence
            label = "%0.02f" % confidence if confidence is not None else None
            self._viewer.color = 0, 0, 255
            self._viewer.rectangle(x, y, w, h, label)

    def draw_track_set(self, frame_idx, track_set):
        track_set_frame = track_set.collect_sensor_data(frame_idx)

        self._viewer.thickness = self.track_set_thickness
        if self.draw_trajectories:
            for tag in track_set_frame.keys():
                self._viewer.color = pymotutils.create_unique_color_uchar(tag)
                track = track_set.tracks[tag]
                for this_frame_idx in range(track.first_frame_idx, frame_idx):
                    if this_frame_idx not in track.detections:
                        continue
                    x, y, w, h = track.detections[this_frame_idx].sensor_data
                    x, y = int(x + w / 2), int(y + h)
                    self._viewer.circle(x, y, 1)
        for tag, (x, y, w, h) in track_set_frame.items():
            self._viewer.color = pymotutils.create_unique_color_uchar(tag)
            self._viewer.rectangle(x, y, w, h, label=str(tag))

    def draw_online_tracking_output(self, tracker):
        self._draw_online_tracking_results(
            self._viewer, self._frame_data, tracker)


def compute_features(detections, image_filenames, feature_extractor):
    """Utility function to pre-compute features.

    Parameters
    ----------
    detections : Dict[int, List[RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections.
    image_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename. The keys of
        the provided detections and image_filenames must match.
    feature_extractor: Callable[ndarray, ndarray] -> ndarray
        The feature extractor takes as input a color image and an Nx4
        dimensional matrix of bounding boxes in format (x, y, w, h) and returns
        an NxM dimensional matrix of N associated feature vectors.

    """

    frame_indices = sorted(list(detections.keys()))
    for frame_idx in frame_indices:
        bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
        assert bgr_image is not None, "Failed to load image"

        rois = np.asarray([d.roi for d in detections[frame_idx]])
        features = feature_extractor(bgr_image, rois)
        for i, feature in enumerate(features):
            setattr(detections[frame_idx][i], "feature", feature)


def extract_image_patches(detections, image_filenames, patch_shape):
    """Utility function to extract image patches of each detetions bounding box.

    On exit, each detection in `detections` has an attribute `image` that
    contains the image patch of shape `patch_shape` that shows the corresponding
    to the bounding box detection.

    Parameters
    ----------
    detections : Dict[int, List[RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections.
    image_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename. The keys of
        the provided detections and image_filenames must match.
    patch_shape : (int, int)
        Image patch shape (width, height). All bounding boxes are reshaped to
        this shape.

    """
    def extract_image_patch(image, bbox):
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, patch_shape[::-1])
        return image

    frame_indices = sorted(list(detections.keys()))
    for frame_idx in frame_indices:
        bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
        assert bgr_image is not None, "Failed to load image"

        rois = np.asarray([d.roi for d in detections[frame_idx]])
        for i, detection in enumerate(detections[frame_idx]):
            setattr(detection, "image", extract_image_patch(bgr_image, rois[i]))
