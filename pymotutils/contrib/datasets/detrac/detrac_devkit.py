# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2

import pymotutils

CAMERA_UPDATE_IN_MS = 33.3   # 30 FPS
CAMERA_IMAGE_SHAPE = (960, 540)


class DataSource(pymotutils.DataSource):
    """
    A data source that provides access to one sequence out of the UA-DETRAC
    dataset.

    Parameters
    ----------
    bgr_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename.
    detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections. Each
        detection contains the bounding box.
    ground_truth : Optional[pymotutils.TrackSet]
        The set of ground-truth tracks.

    Attributes
    ----------
    bgr_filenames : Dict[int, str]
        A dictionary that maps from frame index to image filename.
    detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
        A dictionary that maps from frame index to list of detections. Each
        detection contains the bounding box.
    ground_truth : NoneType | pymotutils.TrackSet
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
        return 25

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
    A development kit for the UA-DETRAC dataset [1]_. To use this development
    kit download the dataset from [1]_. Since the dataset comes in several zip
    files without structure, you can store the extracted files whereever you
    want, but you have to specify the `image_dir`, `detection_dir` and
    `xml_gt_dir`. You can use either the train or test sequences.

    [1]_ http://detrac-db.rit.albany.edu/

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the sequences.
    detection_dir : str
        Path to the directory containing the detections. (Eg. R-CNN, DPM, etc.)
    xml_gt_dir : str
        Path to the directory containing the xml annotations.

    """

    def __init__(self, image_dir, detection_dir, xml_gt_dir):

        self.image_dir = image_dir
        self.detection_dir = detection_dir
        self.xml_gt_dir = xml_gt_dir

    def create_data_source(self, sequence, min_confidence=None):
        """
        Create data source for a given sequence.

        Parameters
        ----------
        sequence : str
            Name of the sequence directory inside the `image_dir`
        min_confidence : Optional[float]
            A detector confidence threshold. All detections with confidence
            lower than this value are disregarded.

        Returns
        -------
        DataSource
            Returns the data source of the given sequence.

        """

        sequence_image_dir = os.path.join(self.image_dir, sequence)

        bgr_filenames = {
            int(os.path.splitext(f[3:])[0]): os.path.join(sequence_image_dir, f)
            for f in sorted(os.listdir(sequence_image_dir))
        }

        detection_dir_name = os.path.basename(
            os.path.normpath(self.detection_dir))
        detection_file = os.path.join(
            self.detection_dir,
            sequence + "_Det_" + detection_dir_name + ".txt")
        detections = pymotutils.motchallenge_io.read_detections(
            detection_file, min_confidence)

        ground_truth_file = os.path.join(self.xml_gt_dir, sequence + '.xml')
        ground_truth = pymotutils.detrac_io.read_groundtruth(
            ground_truth_file)

        return DataSource(bgr_filenames, detections, ground_truth)
