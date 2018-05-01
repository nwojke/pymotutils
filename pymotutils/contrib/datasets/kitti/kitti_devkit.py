# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2

import pymotutils

SEQUENCES_TRAININING = ["%04.d" % i for i in range(1, 21)]
SEQUENCES_TESTING = ["%04.d" % i for i in range(1, 29)]

OBJECT_CLASSES = [
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram",
    "Misc", "DontCare"]

OBJECT_CLASSES_CARS = ["Car"]

OBJECT_CLASSES_PEDESTRIANS = ["Pedestrian"]

MIN_OBJECT_HEIGHT_IN_PIXELS = 25

CAMERA_IMAGE_SHAPE = (1242, 375)
CAMERA_UPDATE_IN_MS = 100  # 10 Hz

GROUND_PLANE_NORMAL = np.array([0., 0., 1.])
GROUND_PLANE_DISTANCE = -0.93


def convert_oxts_to_pose(oxts_list):
    # Converted code from KITTI devkit MATLAB script:
    #
    # Converts a list of oxts measurements into metric poses, starting at
    # (-1,0,0) meters, OXTS coordinates are defined as x = forward, y = right,
    # z = down (see OXTS RT2999 user manual) afterwards, pose[i] contains the
    # transformation which takes a 2D point in the i'th frame and projects it
    # into the oxts coordinates of the first frame.
    def lat_to_scale(lat):
        return np.cos(lat * np.pi / 180.0)

    def lat_lon_to_mercator(lat, lon, scale):
        er = 6378137
        mx = scale * lon * np.pi * er / 180
        my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
        return mx, my

    scale = lat_to_scale(oxts_list[0][0])

    # init pose
    pose_list = []
    inv_transform_0 = None

    # for all oxts packets do
    for i, oxts in enumerate(oxts_list):
        # if there is no data => no pose
        if oxts is None:
            pose_list.append(None)
            continue

        # translation vector
        translation = np.zeros((3, ))
        translation[:2] = lat_lon_to_mercator(oxts[0], oxts[1], scale)
        translation[2] = oxts[2]

        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        rx, ry, rz = oxts[3:6]  # roll, pitch, heading

        # base => nav  (level oxts => rotated oxts)
        rotation_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [
            0, np.sin(rx), np.cos(rx)]])

        # base => nav  (level oxts => rotated oxts)
        rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [
            -np.sin(ry), 0, np.cos(ry)]])

        # base => nav  (level oxts => rotated oxts)
        rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0], [
            np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        # normalize translation and rotation (start at 0/0/0)
        transformation = np.eye(4, 4)
        transformation[:3, :3] = np.linalg.multi_dot(
            (rotation_z, rotation_y, rotation_x))
        transformation[:3, 3] = translation
        if inv_transform_0 is None:
            inv_transform_0 = np.linalg.inv(transformation)

        # add pose
        pose_list.append(np.dot(inv_transform_0, transformation))

    return pose_list


def read_odometry(filename):
    oxts_list = np.loadtxt(filename)
    return convert_oxts_to_pose(oxts_list)


def read_calibration(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()

    data_dict = {}
    for line in lines:
        words = line.strip().split(' ')
        data_dict[words[0].strip(':')] = np.fromstring(
            ";".join(words[1:]), sep=';')

    velodyne_to_camera = np.eye(4)
    velodyne_to_camera[:3, :4] = data_dict["Tr_velo_cam"].reshape(3, 4)

    imu_to_velodyne = np.eye(4)
    imu_to_velodyne[:3, :4] = data_dict["Tr_imu_velo"].reshape(3, 4)
    imu_to_camera = np.dot(velodyne_to_camera, imu_to_velodyne)

    camera_to_rectified = np.eye(4)
    camera_to_rectified[:3, :3] = data_dict["R_rect"].reshape(3, 3)
    imu_to_rectified = np.dot(camera_to_rectified, imu_to_camera)

    projection_matrix = data_dict["P2"].reshape(3, 4)
    return projection_matrix, imu_to_rectified


def read_ground_truth(
        filename, object_classes=None, min_height=MIN_OBJECT_HEIGHT_IN_PIXELS):
    """

    File format:

    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    frame        Frame within the sequence where the object appearers
       1    track id     Unique tracking id of this object within this sequence
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image
                         boundaries.
                         Truncation 2 indicates an ignored object (in particular
                         in the beginning or end of a track) introduced by
                         manual labeling.
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates
                         (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates
                         [-pi..pi]

    """
    with open(filename, "r") as f:
        lines = f.read().splitlines()

    track_set = pymotutils.TrackSet()
    for line in lines:
        words = line.strip().split(' ')
        assert len(words) == 17, "Invalid number of elements in line."
        object_class = words[2]
        if object_class not in object_classes:
            continue
        frame_idx, track_id = int(words[0]), int(words[1])
        roi = np.asarray([float(x) for x in words[6:10]])
        roi[2:] -= roi[:2] - 1  # Convert to x, y, w, h
        if roi[3] < min_height:
            continue

        if track_id not in track_set.tracks:
            track_set.create_track(track_id)
        track_set.tracks[track_id].add(pymotutils.Detection(frame_idx, roi))

    return track_set


def read_detections(
        filename, object_classes=None, min_height=MIN_OBJECT_HEIGHT_IN_PIXELS,
        min_confidence=-np.inf):
    """

    File format:

    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    frame        Frame within the sequence where the object appearers
       1    track id     IGNORED
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    IGNORED
       1    occluded     IGNORED
       1    alpha        IGNORED
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   IGNORED
       3    location     IGNORED
       1    rotation_y   IGNORED
       1    score        Float, indicating confidence in detection, higher is
                         better.

    """
    with open(filename, "r") as f:
        lines = f.read().splitlines()

    detections = {}
    for line in lines:
        words = line.strip().split(' ')
        assert len(words) == 18, "Invalid number of elements in line."
        object_class = words[2]
        if object_class not in object_classes:
            continue
        frame_idx = int(words[0])
        roi = np.asarray([float(x) for x in words[6:10]])
        roi[2:] -= roi[:2] - 1  # Convert to x, y, w, h
        if roi[3] < min_height:
            continue
        confidence = float(words[17])
        if confidence < min_confidence:
            continue
        detections.setdefault(frame_idx, []).append(
            pymotutils.RegionOfInterestDetection(frame_idx, roi, confidence))

    return detections


def write_hypotheses(filename, track_set, object_class):
    lines = []
    for frame_idx in track_set.frame_range():
        track_id_to_bbox = track_set.collect_sensor_data(frame_idx)
        for track_id, bbox in track_id_to_bbox.items():
            line = (
                "%d %d %s -1 -1 -1 %0.2f %0.2f %0.2f %0.2f "
                "-1 -1 -1 -1 -1 -1 -1" % (
                    frame_idx, track_id, object_class, bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3]) + os.linesep)
            lines.append(line)

    with open(filename, "w") as f:
        f.writelines(lines)


class DataSource(pymotutils.DataSource):

    def __init__(
            self, projection_matrix, bgr_filenames, ground_truth, detections,
            sensor_poses, sequence_name, object_classes):
        self.projection_matrix = projection_matrix
        self.bgr_filenames = bgr_filenames
        self.ground_truth = ground_truth
        self.detections = detections
        self.sensor_poses = sensor_poses
        self.sequence_name = sequence_name
        self.object_classes = object_classes

    def apply_nonmaxima_suppression(self, max_bbox_overlap):
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
        return CAMERA_UPDATE_IN_MS

    def read_frame_data(self, frame_idx):
        bgr_image = cv2.imread(self.bgr_filenames[frame_idx], cv2.IMREAD_COLOR)
        frame_data = {
            "bgr_image": bgr_image,
            "detections": self.detections.get(frame_idx, []),
            "ground_truth": self.ground_truth,
            "timestamp": float(frame_idx) * self.update_ms / 1000.,
            "sensor_pose": self.sensor_poses[frame_idx],
            "projection_matrix": self.projection_matrix}
        return frame_data


class Devkit(object):

    def __init__(self, dataset_dir, detection_dir=None):
        # dataset_dir should point to either 'training' or 'testing' dir
        # If detection_dir is not None, takes pickled detections from that
        # directory instead of loading raw the detections from KITTI datset.
        self.dataset_dir = dataset_dir
        self.detection_dir = detection_dir

    def create_data_source(
            self, sequence, object_classes,
            min_height=MIN_OBJECT_HEIGHT_IN_PIXELS, min_confidence=-np.inf):
        image_dir = os.path.join(self.dataset_dir, "image_02", sequence)
        bgr_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))}

        if self.detection_dir is not None:
            detections_filename = os.path.join(
                self.detection_dir, "%s.pkl" % sequence)
            with open(detections_filename, "rb") as f:
                import pickle
                unfiltered_detections = pickle.load(f)

            detections = {}
            for frame_idx, dets in unfiltered_detections.items():
                detections[frame_idx] = [
                    d for d in dets if d.confidence >= min_confidence and
                    d.roi[3] >= min_height]
        else:
            detections_filename = os.path.join(
                self.dataset_dir, "det_02/%s.txt" % sequence)
            detections = read_detections(
                detections_filename, object_classes, min_height,
                min_confidence)

        ground_truth_filename = os.path.join(
            self.dataset_dir, "label_02/%s.txt" % sequence)
        if os.path.exists(ground_truth_filename):
            ground_truth = read_ground_truth(
                ground_truth_filename, object_classes)
        else:
            ground_truth = None

        oxts_filename = os.path.join(
            self.dataset_dir, "oxts/%s.txt" % sequence)
        imu_to_world_list = read_odometry(oxts_filename)

        calibration_filename = os.path.join(
            self.dataset_dir, "calib/%s.txt" % sequence)
        projection_matrix, imu_to_rectified = read_calibration(
            calibration_filename)

        rectified_to_imu = np.linalg.inv(imu_to_rectified)
        frame_idx_to_sensor_pose = {
            i: np.dot(imu_to_world, rectified_to_imu)[:3, :4]
            for i, imu_to_world in enumerate(imu_to_world_list)}

        return DataSource(
            projection_matrix, bgr_filenames, ground_truth, detections,
            frame_idx_to_sensor_pose, sequence, object_classes)
