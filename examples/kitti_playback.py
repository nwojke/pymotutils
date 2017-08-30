# vim: expandtab:ts=4:sw=4
import argparse
import pymotutils
from pymotutils.contrib.datasets import kitti


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="KITTI Dataset Playback")
    parser.add_argument(
        "--kitti_dir", help="Path to KITTI training/testing directory",
        required=True)
    parser.add_argument(
        "--sequence", help="A four digit sequence number", required=True)
    parser.add_argument(
        "--min_confidence",
        help="Detector confidence threshold. All detections with lower "
        "confidence are disregarded", type=float, default=None)
    return parser.parse_args()


def main():
    """Main program entry point."""
    args = parse_args()

    devkit = kitti.Devkit(args.kitti_dir)
    data_source = devkit.create_data_source(
        args.sequence, kitti.OBJECT_CLASSES_PEDESTRIANS,
        min_confidence=args.min_confidence)

    visualization = pymotutils.MonoVisualization(
        update_ms=kitti.CAMERA_UPDATE_IN_MS,
        window_shape=kitti.CAMERA_IMAGE_SHAPE)
    application = pymotutils.Application(data_source)

    # First, play detections. Then, show ground truth tracks.
    application.play_detections(visualization)
    application.play_track_set(data_source.ground_truth, visualization)


if __name__ == "__main__":
    main()
