# vim: expandtab:ts=4:sw=4
import os
import urllib.request
import hashlib
from functools import partial
import tarfile
import glob
import shutil
from xml.dom import minidom
import subprocess

import numpy as np
import cv2

import pymotutils


TRAIN_SEQUENCES = ["S1L1-1", "S1L2-2", "S2L1"]

TEST_SEQUENCES = ["S1L1-2", "S1L2-1", "S2L2", "S2L3"]

CAMERA_IMAGE_SHAPE = (768, 576)
CAMERA_UPDATE_IN_MS = 142.85  # approx. 7 Hz

GROUND_PLANE_NORMAL = np.array([0., 0., 1.])
GROUND_PLANE_DISTANCE = 0.


CROPPED_TRACKING_AREA_MIN = np.array([-14.0696, -14.274, -np.inf])
CROPPED_TRACKING_AREA_MAX = np.array([4.9813, 1.7335, np.inf])


def euler_to_mat(rx, ry, rz, tx, ty, tz, scale=1. / 1000):
    # Default extrinsics are in mm. Use scale 1./1000 for world coordinate frame
    # in meters.
    tx, ty, tz = tx * scale, ty * scale, tz * scale

    # NOTE: this is accoring to the Tsai Camera Calibration Toolbox
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)

    pose = np.eye(4)
    pose[0, 0] = cy * cz
    pose[0, 1] = cz * sx * sy - cx * sz
    pose[0, 2] = sx * sz + cx * cz * sy
    pose[1, 0] = cy * sz
    pose[1, 1] = sx * sy * sz + cx * cz
    pose[1, 2] = cx * sy * sz - cz * sx
    pose[2, 0] = -sy
    pose[2, 1] = cy * sx
    pose[2, 2] = cx * cy

    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


def create_projection_matrix(filename, extrinsic_scale=1. / 1000):
    # Default extrinsics are in mm. Use scale 1./1000 for world coordinate frame
    # in meters.
    xmldoc = minidom.parse(filename)
    geometry = xmldoc.getElementsByTagName("Geometry")[0]
    intrinsic = xmldoc.getElementsByTagName("Intrinsic")[0]
    extrinsic = xmldoc.getElementsByTagName("Extrinsic")[0]

    def g(attr):
        return float(geometry.attributes[attr].value)

    def i(attr):
        return float(intrinsic.attributes[attr].value)

    def e(attr):
        return float(extrinsic.attributes[attr].value)

    world_to_camera = euler_to_mat(
        e("rx"), e("ry"), e("rz"), e("tx"), e("ty"), e("tz"), extrinsic_scale)

    projection_matrix = np.eye(3, 4)
    projection_matrix[0, 0] = i("sx") * i("focal") / g("dpx")
    projection_matrix[1, 1] = i("focal") / g("dpy")
    projection_matrix[0, 2] = i("cx")
    projection_matrix[1, 2] = i("cy")
    return np.dot(projection_matrix, world_to_camera)


def intersect_with_ground_plane(
        inv_projection_matrix, ground_plane_normal, ground_plane_distance,
        points):
    """Find intersection of a ray through an image pixel with the ground plane.

    Plane parameters:
    .. math:: ground_plane_normal.T \cdot x - ground_plane_distance = 0

    Parameters
    ----------
    inv_projection_matrix : ndarray
        The 4x4 inverse of the projection matrix.
    ground_plane_normal : ndarray
        The normal vector of the plane.
    ground_plane_distance : float
        Distance of the plane to origin.
    points : ndarray
        The Nx2 array of pixel coordinates.

    Returns
    -------
    (ndarray, ndarray)
        This method returns the Nx3 array of intersections as well as
        an array of booleans that is True if the intersection point is
        is valid and False if ray and plane are (almost) parallel.

    """
    # 1) Create ray that passes through pixels, transform to world frame
    rays = np.empty((points.shape[0], 4))
    rays[:, :2], rays[:, 2], rays[:, 3] = points, 1., 0.
    rays = np.dot(rays, inv_projection_matrix.T)
    rays /= np.atleast_2d(np.sqrt(np.sum(rays[:, :3] ** 2, axis=1))).T
    rays = rays[:, :3]

    # 2) check for intersection using dot product between rays
    #    and plane normal
    min_dot = np.cos(89 * np.pi / 180.)
    dot_nv = np.sum(rays * ground_plane_normal, axis=1)
    isvalid = np.abs(dot_nv) >= min_dot

    # 3) compute point of intersection
    p = inv_projection_matrix[:3, 3]
    lamda = (ground_plane_distance - np.dot(p, ground_plane_normal)) / dot_nv
    intersection = p + np.atleast_2d(lamda).T * rays
    return intersection, isvalid


def read_cvml_detections(
        filename, projection_matrix, roi_scale_w=0.75, roi_scale_h=1.0):

    def fattr(node, name):
        return float(node.attributes[name].value)

    def rescale_roi(old_roi):
        x, y, w, h = old_roi
        new_w, new_h = roi_scale_w * w, roi_scale_h * h
        dw, dh = w - new_w, h - new_h
        x += dw / 2
        y += dh / 2
        return x, y, new_w, new_h

    wrapped_projection_matrix = np.eye(4)
    wrapped_projection_matrix[:3, :4] = projection_matrix
    inv_projection_matrix = np.linalg.inv(wrapped_projection_matrix)

    xmldoc = minidom.parse(filename)
    detections = {}
    for frame in xmldoc.getElementsByTagName("frame"):
        frame_idx = int(frame.attributes["number"].value)
        detections[frame_idx] = []
        for obj in frame.getElementsByTagName("object"):
            box = obj.getElementsByTagName("box")[0]
            xc, yc = fattr(box, "xc"), fattr(box, "yc")
            w, h = fattr(box, "w"), fattr(box, "h")
            roi = xc - w / 2., yc - h / 2., w, h
            roi = rescale_roi(roi)
            confidence = fattr(obj, "confidence")
            xyz, isvalid = intersect_with_ground_plane(
                inv_projection_matrix, GROUND_PLANE_NORMAL,
                GROUND_PLANE_DISTANCE, np.array([[xc, yc + h / 2.]]))
            assert isvalid[0], "Failed to compute ground plane projection"
            detections[frame_idx].append(
                pymotutils.RegionOfInterestDetection(
                    frame_idx, np.asarray(roi), confidence, xyz[0]))
    return detections


def read_cvml_groundtruth(filename, projection_matrix):

    def fattr(node, name):
        return float(node.attributes[name].value)

    wrapped_projection_matrix = np.eye(4)
    wrapped_projection_matrix[:3, :4] = projection_matrix
    inv_projection_matrix = np.linalg.inv(wrapped_projection_matrix)

    xmldoc = minidom.parse(filename)
    track_set = pymotutils.TrackSet()
    for frame in xmldoc.getElementsByTagName("frame"):
        frame_idx = int(frame.attributes["number"].value)
        for obj in frame.getElementsByTagName("object"):
            box = obj.getElementsByTagName("box")[0]
            xc, yc = fattr(box, "xc"), fattr(box, "yc")
            w, h = fattr(box, "w"), fattr(box, "h")
            roi = xc - w / 2., yc - h / 2., w, h
            xyz, isvalid = intersect_with_ground_plane(
                inv_projection_matrix, GROUND_PLANE_NORMAL,
                GROUND_PLANE_DISTANCE, np.array([[xc, yc + h / 2.]]))
            assert isvalid[0], "Failed to compute ground plane projection"

            track_id = int(obj.attributes["id"].value)
            if track_id not in track_set.tracks:
                track_set.create_track(track_id)
            track_set.tracks[track_id].add(
                pymotutils.RegionOfInterestDetection(
                    frame_idx, roi, xyz=xyz[0]))
    return track_set


def clip_track_set_at_tracking_area(track_set, xyz="sensor_data"):
    cropped_track_set = pymotutils.TrackSet()
    for tag, track in track_set.tracks.items():
        detections = {
            i: d for i, d in track.detections.items()
            if np.all(getattr(d, xyz) >= CROPPED_TRACKING_AREA_MIN)
            and np.all(getattr(d, xyz) <= CROPPED_TRACKING_AREA_MAX)}
        if len(detections) == 0:
            continue
        cropped_track = cropped_track_set.create_track(tag)
        cropped_track.detections = detections
    return cropped_track_set


class DataSource(pymotutils.DataSource):

    def __init__(
            self, projection_matrix, bgr_filenames, ground_truth, detections,
            sequence_name):
        self.projection_matrix = projection_matrix
        self.bgr_filenames = bgr_filenames
        self.ground_truth = ground_truth
        self.detections = detections
        self.sequence_name = sequence_name

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
            "timestamp": float(frame_idx) * CAMERA_UPDATE_IN_MS / 1000.,
            "projection_matrix": self.projection_matrix
        }
        return frame_data


class Devkit(object):

    def __init__(self, dataset_dir):
        self.sequences = TRAIN_SEQUENCES + TEST_SEQUENCES
        self.dataset_dir = dataset_dir

    def download_data(self, base_url=None):
        if base_url is None:
            base_url = "ftp://ftp.cs.rdg.ac.uk/pub/PETS2009/" + \
                       "Crowd_PETS09_dataset/a_data/Crowd_PETS09/"
        print("Download and extract data.")
        self._download_extract_data_if(base_url)
        print("Download and extract calibration.")
        self._download_extract_calibration_if(base_url)
        print("Extracting tracking data.")
        self._download_tracking_data_if()
        print("Done with download and extracting.")

    def create_data_source(
            self, sequence, cropped=False, extrinsic_scale=1. / 1000):
        if sequence not in self.sequences:
            raise KeyError("Unknown sequence '%s'" % sequence)

        projection_matrix = create_projection_matrix(
            os.path.join(self.calibration_dir, "View_001.xml"), extrinsic_scale)

        base_dir = self.get_dataset_dir(sequence)
        image_dir = os.path.join(base_dir, "View_001")
        bgr_filenames = {}
        for filename in os.listdir(image_dir):
            frame_idx = int(filename.replace('.', '_').split('_')[1])
            bgr_filenames[frame_idx] = os.path.join(image_dir, filename)

        data_dir = self.get_tracking_data_dir(sequence)
        if cropped:
            groundtruth_file = os.path.join(
                data_dir, "PETS2009-%s-cropped.xml" % sequence)
        else:
            groundtruth_file = os.path.join(
                data_dir, "PETS2009-%s.xml" % sequence)
        ground_truth = read_cvml_groundtruth(
            groundtruth_file, projection_matrix)

        detections_file = os.path.join(
            data_dir, "PETS2009-%s-c1-det.xml" % sequence)
        detections = read_cvml_detections(detections_file, projection_matrix)

        return DataSource(
            projection_matrix, bgr_filenames, ground_truth, detections,
            sequence)

    @property
    def calibration_dir(self):
        return os.path.join(self.dataset_dir, "Calibration")

    def get_dataset_dir(self, sequence):
        return os.path.join(self.dataset_dir, sequence)

    def get_tracking_data_dir(self, sequence):
        return os.path.join(self.get_dataset_dir(sequence), "Tracking_Data")

    def _download_extract_data_if(self, base_url):
        datasets = [
            "S1_L1.tar.bz2", "S1_L2.tar.bz2", "S1_L3.tar.bz2", "S2_L1.tar.bz2",
            "S2_L2.tar.bz2", "S2_L3.tar.bz2"
        ]
        sha1_sums = [
            "2a15a1f8f81384499081c032ad0ca3bb7e7b88e9",
            "cbd4a825500a4994f1c2ddbf7b4f4dd0ae9493a1",
            "26a26bc7779b88ad9f41b3e672ad44967010176c",
            "ea01601147245f66ea03c82f6b40f98a130441ed",
            "c1aaf3559ba758bee68aa572b798ff64a0eeb076",
            "7be2e22b4d8fa44186c4bcfd26eb32e7d299cd72"
        ]

        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        for dataset, sha1_sum in zip(datasets, sha1_sums):
            url = os.path.join(base_url, dataset)
            filename = os.path.join(self.dataset_dir, dataset)
            self._download_if(url, filename, sha1_sum)
            self._extract_if(filename)

    def _download_extract_calibration_if(self, base_url):
        if os.path.isdir(self.calibration_dir):
            return
        calibration_file = "Calibrationxmls.tar"
        calibration_sha1 = "8d1d21a5e832f751150a57c23716bb39dc70043c"
        self._download_if(
            os.path.join(base_url, calibration_file),
            os.path.join(self.dataset_dir, calibration_file), calibration_sha1)

        print("Extracting calibration")
        tar = tarfile.open(os.path.join(self.dataset_dir, calibration_file))
        tar.extractall(self.calibration_dir)
        tar.close()
        print("Patching XML files")
        for filename in os.listdir(self.calibration_dir):
            filename = os.path.join(self.calibration_dir, filename)
            subprocess.call(["sed", "-i", "s/dpx\"/dpx=\"/g", filename])
            subprocess.call(["sed", "-i", "s/dpy\"/dpy=\"/g", filename])
        print("Done.")

    def _download_tracking_data_if(self):

        def download_gt_if(seq, filename):
            path = os.path.join(self.get_tracking_data_dir(seq), filename)
            if os.path.isfile(path):
                return
            base_url = "http://www.milanton.de/files/gt/PETS2009/"
            print("Downloading %s" % os.path.join(base_url, filename))
            urllib.request.urlretrieve(os.path.join(base_url, filename), path)
            print("Done.")

        def download_det_if(seq, filename):
            path = os.path.join(self.get_tracking_data_dir(seq), filename)
            if os.path.isfile(path):
                return
            base_url = "http://www.milanton.de/files/det/PETS2009/"
            print("Downloading %s" % os.path.join(base_url, filename))
            urllib.request.urlretrieve(os.path.join(base_url, filename), path)
            print("Done.")

        for sequence in self.sequences:
            os.makedirs(self.get_tracking_data_dir(sequence), exist_ok=True)
            download_gt_if(sequence, "PETS2009-%s.xml" % sequence)
            download_gt_if(sequence, "PETS2009-%s-cropped.xml" % sequence)
            download_det_if(sequence, "PETS2009-%s-c1-det.xml" % sequence)

    def _download_if(self, url, filename, sha1_sum):
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                d = hashlib.sha1()
                for buf in iter(partial(file.read, 128), b''):
                    d.update(buf)
                if d.hexdigest() == sha1_sum:
                    return
        print("Downloading %s" % url)
        urllib.request.urlretrieve(
            url, os.path.join(self.dataset_dir, filename))
        print("Done.")

    def _extract_if(self, filename):
        print("Extracting %s" % filename)
        tmpdir = os.path.join(self.dataset_dir, "tmp")
        tar = tarfile.open(filename)
        tar.extractall(tmpdir)
        tar.close()

        # get destination directory
        dataset_dir = os.path.basename(filename).replace("_", "").split(".")[0]
        dest_dir = os.path.join(self.dataset_dir, dataset_dir)
        if os.path.isdir(dataset_dir) or len(
                glob.glob(dataset_dir + "-*")) > 0:
            # glob, because of possible suffix, e.g., S1_L1-1
            return

        # get directory of extracted data
        scontainer = os.path.join(tmpdir, "Crowd_PETS09")
        spath = glob.glob(os.path.join(scontainer, "*"))[0]
        sname = os.path.basename(spath)

        lcontainer = os.path.join(scontainer, sname)
        lpath = glob.glob(os.path.join(lcontainer, "*"))[0]
        lname = os.path.basename(lpath)

        tcontainer = os.path.join(lcontainer, lname)
        tnames = glob.glob(os.path.join(tcontainer, "*"))

        print("Copying files")
        for i, tname in enumerate(sorted(tnames)):
            if len(tnames) > 1:
                dest_dir += "-%d" % (1 + i)
            views = sorted(os.listdir(os.path.join(tcontainer, tname)))
            for view in views:
                source_dir = os.path.join(tcontainer, tname, view)
                os.makedirs(dest_dir, exist_ok=True)

                filenames = glob.glob(os.path.join(source_dir, "*.jpg"))
                for filename in filenames:
                    shutil.copyfile(
                        filename,
                        os.path.join(dest_dir, os.path.basename(filename)))

        print("Removing temporary files")
        shutil.rmtree(tmpdir)
        print("Done.")
