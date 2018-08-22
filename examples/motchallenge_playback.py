# vim: expandtab:ts=4:sw=4
import argparse
import pymotutils
from pymotutils.contrib.datasets import motchallenge

TRAJECTORY_VISUALIZATION_LEN_IN_MSECS = 3000.0

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MOTChallenge Dataset Playback")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge train/test directory",
        required=True)
    parser.add_argument(
        "--sequence", help="Name of the sequence to play", required=True)
    parser.add_argument(
        "--min_confidence",
        help="Detector confidence threshold. All detections with lower "
        "confidence are disregarded", type=float, default=None)
    return parser.parse_args()


def main():
    """Main program entry point."""
    args = parse_args()

    devkit = motchallenge.Devkit(args.mot_dir)
    data_source = devkit.create_data_source(args.sequence, args.min_confidence)

    # Compute a suitable window shape.
    image_shape = data_source.peek_image_shape()[::-1]
    aspect_ratio = float(image_shape[0]) / image_shape[1]
    window_shape = int(aspect_ratio * 600), 600

    visualization = pymotutils.MonoVisualization(
        update_ms=data_source.update_ms, window_shape=window_shape)
    visualization.trajectory_visualization_len = int(
        TRAJECTORY_VISUALIZATION_LEN_IN_MSECS / data_source.update_ms)
    application = pymotutils.Application(data_source)

    # First, play detections. Then, show ground truth tracks.
    application.play_detections(visualization)
    application.play_track_set(data_source.ground_truth, visualization)


if __name__ == "__main__":
    main()
