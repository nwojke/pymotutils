# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2

import pymotutils


class DataSource(pymotutils.DataSource):
    """
    A data source that provides access to one sequence out of the MOTChallenge
    dataset.

    Parameters
    ----------
    bgr_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename.
    detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections. Each
        detection contains the bounding box and, if provided, the 3D object
        coordinates (via attribute `xyz`).
    ground_truth : Optional[TrackSet]
        The set of ground-truth tracks.

    Attributes
    ----------
    bgr_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename.
    detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections. Each
        detection contains the bounding box and, if provided, the 3D object
        coordinates (via attribute `xyz`).
    ground_truth : NoneType | TrackSet
        The set of ground-truth tracks, if available.

    """

    def __init__(self, bgr_filenames, detections, ground_truth=None):
        self.bgr_filenames = bgr_filenames
        self.detections = detections
        self.ground_truth = ground_truth

    def apply_nonmaxima_suppression(self, max_bbox_overlap):
        """Apply non-maxima suppression.

        Parameters
        ----------
        max_bbox_overlap : float
            ROIs that overlap more than this value are suppressed.

        Returns
        -------

        """
        for frame_idx, detections in self.detections.items():
            if len(detections) == 0:
                continue
            boxes = np.asarray([d.roi for d in detections])
            scores = np.asarray([d.confidence for d in detections])
            indices = pymotutils.preprocessing.non_max_suppression(
                boxes, max_bbox_overlap, scores)
            self.detections[frame_idx] = [detections[i] for i in indices]

    def first_frame_idx(self):
        return min(self.bgr_filenames.keys())

    def last_frame_idx(self):
        return max(self.bgr_filenames.keys())

    @property
    def update_ms(self):
        return 25  # TODO(nwojke): Peek correct frame rate from seqinfo.ini file

    def read_frame_data(self, frame_idx):
        bgr_image = cv2.imread(self.bgr_filenames[frame_idx], cv2.IMREAD_COLOR)
        frame_data = {
            "bgr_image": bgr_image,
            "detections": self.detections.get(frame_idx, []),
            "ground_truth": self.ground_truth,
            "timestamp": float(frame_idx)
        }
        return frame_data

    def peek_image_shape(self):
        """Get the image shape for this sequence in format (height, width). """
        image = cv2.imread(next(iter(self.bgr_filenames.values())))
        return image.shape[:2]


class Devkit(object):
    """
    A development kit for the MOTChallenge dataset [1]_. To use this development
    kit, download the dataset from [1]_ and set the `dataset_dir` to either
    the train or test directory. Then, create a DataSource for one of the
    sequences contained in this directory.

    [1]_ http://www.motchallenge.net

    Parameters
    ----------
    dataset_dir : str
        Path to the MOTChallenge train or test directory.
    detection_dir : Optional[str]
        Optional path to a directory containing custom detections. The expected
        filename is `detection_dir/[sequence_name].txt`. Detections must be
        stored in the original MOTChallenge format.

    Attributes
    ----------
    dataset_dir : str
        Path to the MOTChallenge train/test directory.
    detection_dir : NoneType | str
        If not None, a path to a directory containing custom detections in
        MOTChallenge format.

    """

    def __init__(self, dataset_dir, detection_dir=None):
        self.dataset_dir = dataset_dir
        self.detection_dir = detection_dir

    def create_data_source(self, sequence, min_confidence=None):
        """Create data source for a given sequence.

        Parameters
        ----------
        sequence : str
            Name of the sequence directory inside the `dataset_dir`.
        min_confidence : Optional[float]
            A detector confidence threshold. All detections with confidence
            lower than this value are disregarded.

        Returns
        -------
        DataSource
            Returns the data source of the given sequence.

        """
        sequence_dir = os.path.join(self.dataset_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        bgr_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
        }

        detection_file = (
            os.path.join(sequence_dir, "det", "det.txt")
            if self.detection_dir is None
            else os.path.join(self.detection_dir, "%s.txt" % sequence))
        detections = pymotutils.motchallenge_io.read_detections(
            detection_file, min_confidence)

        ground_truth_file = os.path.join(sequence_dir, "gt", "gt.txt")
        ground_truth = pymotutils.motchallenge_io.read_groundtruth(
            ground_truth_file, sensor_data_is_3d=False)  # Evaluation always 2D

        # TODO(nwojke): MOT16 and newer have a seqinfo.ini file that contains
        # information on the frame rate and image size.
        return DataSource(bgr_filenames, detections, ground_truth)
