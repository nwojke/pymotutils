# vim: expandtab:ts=4:sw=4
"""
This module contains base classes for tracking applications that allow for
easy integration of new datasets and tracking algorithms. The application
structure is divided into data acquisition, tracking, visualization, and
application glue code.

The modules application base class performs much of the often replicated
functionality for running state estimation, data association, and evaluation.
The module also contains abstract base classes that define the interface for
data acquisition, tracking, and visualization.
"""
import six
import abc
import time
import pymotutils


@six.add_metaclass(abc.ABCMeta)
class DataSource(object):
    """
    This is an abstract base class that defines the interface between any data
    sources, e.g., a public dataset, and the application base class provided by
    this module.

    Attributes
    ----------
    """

    @abc.abstractmethod
    def first_frame_idx(self):
        """Get index of the first frame (usually 0).

        Returns
        -------
        int
            Index of the first frame.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def last_frame_idx(self):
        """Get index of the last frame or None, if data source has no defined
        end.

        Returns
        -------
        int | NoneType
            Index of the last frame or None.

        """
        raise NotImplementedError("abstract base class")

    def num_frames(self):
        """Get number of frames in the data source or None, if there is no
        defined end.

        Returns
        -------
        int
            Number of frames in the data source or None.

        """
        if self.last_frame_idx() is None:
            return None
        else:
            return self.last_frame_idx() - self.first_frame_idx() + 1

    @abc.abstractmethod
    def read_frame_data(self, frame_idx):
        """Read a given frame into memory.

        This method is called by the application base class to read data of a
        particular frame into memory. The data is returned as a dictionary
        that must contain all data that is necessary for visualization
        of tracking results (see :class:`Visualization`) and at least the
        following::

        * "detections": The set of detections at the given time step. This is
          passed on to the tracker.
        * "timestamp": The timestamp of the given frame. This is used to
          configure the trackers motion model.

        Optionally, the frame data may contain::

        * "sensor_pose": sensor pose at the current time step, this is
          passed on to the tracker. By convention, this should be an affine
          transformation matrix.
        * "ground_truth": ground truth data over the entire sequence, i.e., a
          :class:`TrackSet` that contains the multi-target ground truth
          trajectory of the entire sequence. This item should contain the full
          track set (over the entire sequence) for all given frame_idx.

        By convention, we currently use the following attribute names for
        visualization-dependent data::

        * "bgr_image": a single color image in BGR color space
        * "bgr_images": color images of multiple sensors, all in BGR color space
        * "disparity_image": a single disparity image
        * "disparity_images": disparity images of multiple sensors

        Parameters
        ----------
        frame_idx : int
            The index of the frame to load.

        Returns
        -------
        Dict[str, T]
            This method returns a dictionary of frame-dependent data, such
            as timestamps, detections, etc. See description above.

        """
        raise NotImplementedError("abstract base class")


@six.add_metaclass(abc.ABCMeta)
class Visualization(object):
    """
    This is an abstract class that defines the interface between the modules
    application base class and visualization of tracking results.

    During visualization, the control flow is handed over from the application
    to the visualization object. Therefore, every concrete implementation of
    this class must provide a control loop that iterates over the entire
    sequence of data.
    """

    @abc.abstractmethod
    def run(self, start_idx, end_idx, frame_callback):
        """Run visualization between a given range of frames.

        This method is called by the application base class. At this point, the
        application hands over control to the visualization, which is expected
        to call the given callback at each frame. From within the callback, the
        application will call the visualization routines declared in this
        class.

        Parameters
        ----------
        start_idx : int
            The index of the first frame to show.
        end_idx : Optional[int]
            One plus the index of the last frame to show. If None given, the
            visualization should run forever or until the user has requested
            to terminate.
        frame_callback : Callable[int] -> None
            A callback that must be invoked at each frame from start_idx to
            end_idx - 1. As argument, the index of the current frame should
            be passed.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def init_frame(self, frame_data):
        """Initialize visualization routines.

        This method is called once at the beginning of each frame.

        .. see: `class:`DataSource`

        Parameters
        ----------
        frame_data : Dict[str, T]
            The dictionary of frame-dependent data. See :class:`DataSource`
            for more information.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def finalize_frame(self):
        """Finalize visualization routines.

        This method is called once at the end of each frame.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def draw_detections(self, detections):
        """Draw detections at the current time step.

        Parameters
        ----------
        detections : List[Detection]
            The set of detections at the current time step. The concrete type
            is application specific.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def draw_online_tracking_output(self, tracker):
        """Draw online tracking results.

        Called once every frame after all processing has been done.

        Parameters
        ----------
        tracker : Tracker
            The multi-target tracker.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def draw_track_set(self, frame_idx, track_set):
        """Draw a set of tracks at the current time step.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame that should be visualized.
        track_set : TrackSet
            The set of tracks to visualize. The concrete type of sensor data
            that is contained in the track set is application dependent.

        """
        raise NotImplementedError("abstract base class")


class NoVisualization(Visualization):
    """
    A simple visualization object that loops through the sequence without
    showing any results.
    """

    def run(self, start_idx, end_idx, frame_callback):
        """Initiate control loop for the given number of frames.

        Parameters
        ----------
        start_idx : int
            Index of the first frame to process.
        end_idx : Optional[int]
            One plus the index of the last frame to process. If None given,
            control is executed in an endless loop.
        frame_callback : Callable[int] -> None
            A callable that is invoked at each frame. As argument, the index
            of the current frame is passed.

        """
        for frame_idx in range(start_idx, end_idx):
            print("Frame index: %d / %d" % (frame_idx, end_idx))
            frame_callback(frame_idx)

    def init_frame(self, frame_data):
        pass

    def finalize_frame(self):
        pass

    def draw_detections(self, detections):
        pass

    def draw_online_tracking_output(self, tracker):
        pass

    def draw_track_set(self, frame_idx, track_set):
        pass


@six.add_metaclass(abc.ABCMeta)
class Tracker(object):
    """
    This is the abstract base class of tracking algorithms. The class defines
    a common interface that is enforced by the application base class.

    You can assume that data is processed sequentially, one frame at a time.

    """

    @abc.abstractmethod
    def reset(self, start_idx, end_idx):
        """Reset the tracker.

        This method is called once before processing the data to inform the
        tracker that a new sequence will be processed.

        Parameters
        ----------
        start_idx : int
            Index of the first frame of upcoming sequence.
        end_idx : Optional[int]
            One plus index of the last frame of the upcoming sequence, or None
            if there is no predefined end.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def process_frame(self, frame_data):
        """Process incoming detections of a new frame.

        This method is called once for every frame, sequentially.

        Parameters
        ----------
        frame_data : Dict[str, T]
            The dictionary of frame-dependent data. See :class:`DataSource`
            for more information. Must include a `timestamp`, a list of
            `detections` and optionally `sensor_pose`.

        Returns
        -------
        Dict[int, tracking.track_hypothesis.TrackHypothesis] | NoneType
            Returns a dictionary that maps from track identity to track
            hypothesis, or None if identities cannot be resolved online.

        """
        raise NotImplementedError("abstract base class")

    @abc.abstractmethod
    def compute_trajectories(self):
        """Compute trajectories.

        This method is called once at the end of the sequence to obtain
        target trajectories.

        Parameters
        ----------

        Returns
        -------
        List[List[Detection]]
            A list of target trajectories, where each target trajectory is
            a list of detections that belong to the same object.

            The sensor_data field in each detection should be compatible with
            the concrete implementation of track set visualization.

        """
        raise NotImplementedError("abstract base class")


class Application(object):
    """
    This is the application base class that provides functionality for running
    a (multi-)target tracker on a particular data source and for evaluation
    against ground truth data.

    Parameters
    ----------
    data_source : DataSource
        The concrete data source of the experiment.

    Attributes
    ----------
    data_source : DataSource
        The data source of this application.
    hypotheses : TrackSet
        Track hypotheses recorded during last execution.
    ground_truth : TrackSet
        Ground truth tracks recorded during last execution.

    """

    def __init__(self, data_source):
        assert isinstance(
            data_source, DataSource), "data_source is of wrong type"
        self.data_source = data_source
        self.hypotheses = pymotutils.TrackSet()
        self.ground_truth = pymotutils.TrackSet()

        self._visualization = None
        self._playback_trackset = pymotutils.TrackSet()
        self._tracker = None
        self._prev_timestep = None

    def play_track_set(
            self, track_set, visualization, start_idx=None, end_idx=None):
        """Loop through dataset and visualize a given track set.

        This method calls visualization routines for drawing the detections
        contained in the given track set.

        Parameters
        ----------
        track_set : TrackSet
            The set of tracks to visualize.
        visualization : Visualization
            A concrete implementation of Visualization that draws the track set.
        start_idx : Optional[int]
            Index of the first frame. Defaults to 0.
        end_idx : Optional[int]
            One plus index of the last frame. Defaults to the number of frames
            in the data source.

        """
        assert isinstance(
            visualization, Visualization), "visualization is of wrong type"
        start_idx = (
            start_idx
            if start_idx is not None else self.data_source.first_frame_idx())

        source_end_idx = (
            self.data_source.last_frame_idx() + 1
            if self.data_source.last_frame_idx() is not None else None)
        end_idx = (end_idx if end_idx is not None else source_end_idx)

        self._visualization = visualization
        self._playback_trackset = track_set
        visualization.run(start_idx, end_idx, self._next_frame_playback)

    def play_groundtruth(self, visualization, start_idx=None, end_idx=None):
        """Play ground truth.

        This method visualizes the ground truth data that has been collected
        during the last run of process_data. If process_data has not been
        executed before calling this function, the ground truth will be empty.

        If you want to play the full ground truth data without evaluating
        a tracker, you can call play_track_set on data contained in the
        data source::

            >>> ground_truth = my_data_source.read_frame_data(0)["ground_truth"]
            >>> app = Application(my_data_source)
            >>> app.play_track_set(ground_truth)

        This works, because by convention the ground_truth returned for a
        particular frame always contains the etire track set.

        Parameters
        ----------
        visualization : Visualization
            A concrete implementation of Visualization that draws the track set.
        start_idx : Optional[int]
            Index of the first frame. Defaults to 0.
        end_idx : Optional[int]
            One plus index of the last frame. Defaults to the number of frames
            in the data source.

        """
        self.play_track_set(
            self.ground_truth, visualization, start_idx, end_idx)

    def play_hypotheses(self, visualization, start_idx=None, end_idx=None):
        """Play tracking results.

        This method visualizes the tracking results that has been collected
        during the last run of process_data. If process_data has not been
        executed before calling this function, the tracking results will be
        empty.

        Parameters
        ----------
        visualization : Visualization
            A concrete implementation of Visualization that draws the track set.
        start_idx : Optional[int]
            Index of the first frame. Defaults to 0.
        end_idx : Optional[int]
            One plus index of the last frame. Defaults to the number of frames
            in the data source.

        """
        self.play_track_set(self.hypotheses, visualization, start_idx, end_idx)

    def _next_frame_playback(self, frame_idx):
        frame_data = self.data_source.read_frame_data(frame_idx)
        self._visualization.init_frame(frame_data)
        self._visualization.draw_track_set(frame_idx, self._playback_trackset)
        self._visualization.finalize_frame()

    def play_detections(self, visualization, start_idx=None, end_idx=None):
        """Show detections.

        Parameters
        ----------
        visualization : Visualization
            A concrete implementation of Visualization that draws the
            detections.
        start_idx : Optional[int]
            Index of the first frame. Defaults to 0.
        end_idx
            One plus index of the last frame. Defauls to the number of frames
            in the data source.

        """
        assert isinstance(
            visualization, Visualization), "visualization is of wrong type"
        start_idx = (
            start_idx
            if start_idx is not None else self.data_source.first_frame_idx())

        source_end_idx = (
            self.data_source.last_frame_idx() + 1
            if self.data_source.last_frame_idx() is not None else None)
        end_idx = (end_idx if end_idx is not None else source_end_idx)

        self._visualization = visualization
        visualization.run(start_idx, end_idx, self._next_frame_detections)

    def _next_frame_detections(self, frame_idx):
        frame_data = self.data_source.read_frame_data(frame_idx)
        self._visualization.init_frame(frame_data)
        self._visualization.draw_detections(frame_data["detections"])
        self._visualization.finalize_frame()

    def process_data(
            self, tracker, visualization=None, start_idx=None, end_idx=None):
        """Process a batch of frames.

        This method runs the given tracker on a sequence of data and collects
        the ground truth detections contained within.

        Parameters
        ----------
        tracker : Tracker
            A concrete tracking implementation.
        visualization : Visualization
            A concrete implementation of Visualization that draws detections
            and state estimates.
        start_idx : Optional[int]
            Index of the first frame. Defaults to 0.
        end_idx : Optional[int]
            One plus index of the last frame. Defaults to the number of frames
            in the data source.

        """
        if visualization is None:
            visualization = NoVisualization()
        assert isinstance(
            visualization, Visualization), "visualization is of wrong type"
        assert isinstance(tracker, Tracker), "tracker is of wrong type"

        start_idx = (
            start_idx
            if start_idx is not None else self.data_source.first_frame_idx())
        end_idx = (
            end_idx
            if end_idx is not None else self.data_source.last_frame_idx() + 1)

        self._visualization = visualization
        self.ground_truth = pymotutils.TrackSet()
        self.hypotheses = pymotutils.TrackSet()
        self._tracker = tracker

        self._tracker.reset(start_idx, end_idx)
        self._visualization.run(
            start_idx, end_idx, self._next_frame_process_data)

    def _next_frame_process_data(self, frame_idx):
        frame_data = self.data_source.read_frame_data(frame_idx)
        detections = frame_data["detections"]

        t0 = time.time()
        self._tracker.process_frame(frame_data)
        t1 = time.time()
        print("Processing time for this frame:", 1e3 * (t1 - t0), "ms")

        self._visualization.init_frame(frame_data)
        self._visualization.draw_detections(detections)
        self._visualization.draw_online_tracking_output(self._tracker)
        self._visualization.finalize_frame()

    def compute_trajectories(self, interpolation, detection_converter=None):
        """Compute trajectories on the sequence of data that has previously been
        processed.

        You must call process_data before computing trajectories, otherwise
        this method will fail.

        Optionally, you can pass in a function for interpolating tracking
        results. Prior to interpolation, you may convert tracking results
        using a user-specified function.

        See
        ----
        interpolate_track_set
        convert_track_set

        Parameters
        ----------
        interpolation : bool | Callable[TrackSet] -> TrackSet
            If True, track hypotheses and ground truth data will be
            interpolated (i.e., missed detections will be filled).
            If False, track hypotheses and ground truth will not be
            interpolated.
            Alternatively, you can pass in a function to use for track set
            interpolation.
        detection_converter : Optional[Callable[Detection] -> Detection]
            A converter function that is called once for each detection in order
            to convert tracking results to a format that is suitable for
            evaluation. This function is called prior to interpolation.

        Returns
        -------
        List[List[Detection]]
            Returns a list of trajectories, where each trajectory is a
            sequence of detections that belong to the same object.

        """
        trajectories = self._tracker.compute_trajectories()
        self.hypotheses = pymotutils.TrackSet()
        for i, trajectory in enumerate(trajectories):
            track = self.hypotheses.create_track(i)
            for detection in trajectory:
                track.add(detection)

        if detection_converter is not None:
            self.hypotheses = pymotutils.postprocessing.convert_track_set(
                self.hypotheses, detection_converter)

        if not isinstance(interpolation, bool):
            self.ground_truth = interpolation(self.ground_truth)
            self.hypotheses = interpolation(self.hypotheses)
        elif interpolation:
            interpolation = pymotutils.postprocessing.interpolate_track_set
            self.ground_truth = interpolation(self.ground_truth)
            self.hypotheses = interpolation(self.hypotheses)

        return trajectories
