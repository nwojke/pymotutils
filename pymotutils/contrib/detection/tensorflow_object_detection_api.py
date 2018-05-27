# vim: expandtab:ts=4:sw=4
"""
This module contains code to run models from the TensorFlow detection model
zoo [1].

[1] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
"""
import cv2
import numpy as np
import tensorflow as tf
import pymotutils


class Detector(object):
    """
    A thin wrapper class around the TensorFlow Object Detection inference
    API.

    Parameters
    ----------
    inference_graph_pb : str
        Path to the frozen_inference_graph.pb file. This file is contained in
        the model archive.

    """

    def __init__(self, inference_graph_pb):
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(inference_graph_pb, "rb") as file_handle:
                serialized_graph = file_handle.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name="")

            self._image_tensor = self._detection_graph.get_tensor_by_name(
                "image_tensor:0")
            self._detection_boxes = self._detection_graph.get_tensor_by_name(
                "detection_boxes:0")
            self._detection_scores = self._detection_graph.get_tensor_by_name(
                "detection_scores:0")
            self._detection_classes = self._detection_graph.get_tensor_by_name(
                "detection_classes:0")
        self._session = tf.Session(graph=self._detection_graph)

    def run(self, bgr_image, min_confidence=0.5, max_bbox_overlap=0.7):
        """Run object detector on single image.

        Parameter
        ---------
        bgr_image : ndarray
            Input image in BGR color space.
        min_confidence : float
            Minimum detector confidence in [0, 1]. Detections with confidence
            lower than this value are suppressed.
        max_bbox_overlap : float
            Non-maxima suppression threshold in [0, 1]. A large value
            reduces the number of returned detections.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            Returns a tuple containing the following elements:
            * An array of shape (N, 4) which contains the bounding boxes of
              N object detections in format (top-left-x, top-left-y, width,
              height).
            * An array of shape (N, ) which contains the corresponding detector
              confidence score.
            * An array of shape (N, ) which contains the corresponding class
             label (integer-valued).

        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        with self._detection_graph.as_default():
            boxes, scores, classes = self._session.run([
                self._detection_boxes, self._detection_scores,
                self._detection_classes], feed_dict={
                    self._image_tensor: rgb_image[np.newaxis, :, :, :]})
            boxes, scores, classes = (
                boxes[0], scores[0], classes[0].astype(np.int32))

        keep = np.greater_equal(scores, min_confidence)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        # Convert to (x, y, width, height).
        boxes[:, :2] *= np.asarray(bgr_image.shape[:2])
        boxes[:, 2:] *= np.asarray(bgr_image.shape[:2])
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

        keep = pymotutils.preprocessing.non_max_suppression(
            boxes, max_bbox_overlap, scores)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        return boxes, scores, classes


def generate_detections(
        index_to_bgr_filenames, inference_graph_pb, class_to_name,
        min_confidence=0.5, max_bbox_overlap=0.7, verbose=False):
    """Generate detections from list of image filenames.

    Parameters
    ----------
    index_to_bgr_filenames: Dict[int, str]
        Maps from frame index to image filename. The frame index is used to
        populate the RegionOfInterestDetection.frame_idx attribute.
    inference_graph_pb : str
        Path to the frozen_inference_graph.pb file. This file is contained in
        the model archive.
    class_to_name : Dict[int, str]
        A dictionary that maps from label to class name. Classes that are not
        contained in the dictionary are suppressed. Use MSCOCO_LABELMAP for
        networks trained on MSCOCO.
    min_confidence : float
        Minimum detector confidence in [0, 1]. Detections with confidence
        lower than this value are suppressed.
    max_bbox_overlap : float
        Non-maxima suppression threshold in [0, 1]. A large value
        reduces the number of returned detections.
    verbose : bool
        If True, prints status information about the number of processed frames
        to standard output.

    Returns
    -------
    Dict[int, List[pymotutils.RegionOfInterestDetection]]
        Returns a dictionary that maps from frame index to list of detections.

    """
    detector = Detector(inference_graph_pb)
    detections = dict()

    num_processed = 0
    for frame_idx, filename in sorted(list(index_to_bgr_filenames.items())):
        if verbose:
            print(
                "Processing detection on frame %d out of %d" %
                (num_processed, len(index_to_bgr_filenames)))
            num_processed += 1
        bgr_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        boxes, scores, classes = detector.run(
            bgr_image, min_confidence, max_bbox_overlap)

        keep = [i for i in range(len(boxes)) if classes[i] in class_to_name]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        class_names = [class_to_name[x] for x in classes]

        detections[frame_idx] = {
            pymotutils.RegionOfInterestDetection(
                frame_idx, boxes[i], scores[i], class_label=classes[i],
                class_name=class_names[i]) for i in range(len(boxes))}
    return detections


"""
This dictionary provides the mapping from class ID to display_name for networks
trained on MSCOCO.
"""
MSCOCO_LABELMAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"}

"""
This dictionary provides the mapping from class ID to display_name for networks
trained on KITTI.
"""
KITTI_LABELMAP = { 1: "car", 2: "pedestrian" }
