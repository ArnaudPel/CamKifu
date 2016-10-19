import importlib
import os.path
import re
import threading
import time
from os import listdir

import cv2

import camkifu.core.video
from camkifu.config import cvconf
from camkifu.config.cvconf import snapshot_dir
from camkifu.core.imgutil import is_img, get_video_capture


class VManagerBase(threading.Thread):
    """ Abstract vision manager, responsible for creating and coordinating video processors.

    Attributes:
        controller: Controller
            A controller that can handle instructions coming from vision ('append', 'delete', 'bulk')
        imqueue: Queue
            Contains the images to be displayed.

        capt: VideoCapture
            Responsible for handling the reading of frames from video input.
        current_video: int or str
            The current video input descriptor, can be an int to define a device, or a file path to a video.
            See cv2.VideoCapture()

        bf_class: type
            The current board finder class that should be used to detect the Goban on the global images.
        sf_class: type
            The current stones finder class that should be used to detect the stones from the canonical (Goban) image.
        board_finder: BoardFinder
            The current video processor instance responsible for board detection, should be of type 'bf_class'.
        stones_finder: StonesFinder
            The current video processor instance responsible for stones detection, should be of type 'sf_class'.

        full_speed: bool
            Indicate whether the video processor should run at full speed. Typically this would be True for files, and
            False for live input.
    """

    def __init__(self, controller, imqueue=None, bf=None, sf=None):
        threading.Thread.__init__(self, name="Vision")
        self.controller = controller
        self.punch_controller()
        self.imqueue = imqueue

        self.capt = None  # initialized in run() with video argument
        self.current_video = None

        self.bf_class = self._reflect(bf, cvconf.bfinders)
        self.sf_class = self._reflect(sf, cvconf.sfinders)
        self.board_finder = None
        self.stones_finder = None

        self.full_speed = False

    def punch_controller(self):
        """ Dynamic setting of a few controller methods.

        Some of the methods called from inside the controller are actually implemented by this VManager instance.
        This practice is most likely not the cleanest way to handle it, but it'll have to do for now.
        """
        self.controller.corrected = self.corrected
        self.controller.next = self.next

    def init_capt(self):
        """ Initialize or reset the video capture object with video input as defined in the controller.
        """
        self.current_video = self.controller.video
        if self.capt is not None:
            self.capt.release()
        if is_img(self.controller.video):
            self.capt = CaptureReaderImg(self.controller.video)
        else:
            self.capt = self._get_capture()

        if self.capt is not None:
            self.full_speed = os.path.isfile(self.controller.video)
            # set the reading position in video files. is ignored by live camera
            self.capt.set(cv2.CAP_PROP_POS_AVI_RATIO, self.controller.bounds[0])

    def _get_capture(self):
        """ Return the proper video capture object for this vmanager (may be a wrapper of cv2.VideoCapture).
        """
        # noinspection PyArgumentList
        return CaptureReaderBase(cv2.VideoCapture(self.controller.video), self)

    def run(self):
        """ Start and manage vision processes.
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def interrupt(self):
        """ Request the interruption of self.run().
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def stop_processing(self):
        """ Request this video manager to terminate all its video-processing.

        self.run() may keep a main loop running, allowing processing to be restarted later on.
        See also self.interrupt() for a more definitive stop.
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def error_raised(self, processor, error):
        """ Print the error and interrupt immediately.

        Args:
            processor:VidProcessor
                Processor from which the error originated.
        """
        vname = self.__class__.__name__
        pname = processor.__class__.__name__
        ename = error.__class__.__name__
        print("{} terminating due to {} in {}.".format(vname, ename, pname))
        self.interrupt()

    def read(self, caller):
        """ Read the next frame from the video input.
        """
        return self.capt.read(caller)

    def next(self):
        """ Call next() on all VidProcessors.
        """
        if self.board_finder is not None:
            self.board_finder.next()
        if self.stones_finder is not None:
            self.stones_finder.next()

    def corrected(self, err_move, exp_move):
        """ Register a correction made by the user on the goban.

        There is no need to perform any direct action, but this information may be used to tune detection algorithms.
        One major aspect is to prevent deleted stones to be re-suggested right after the user has removed them.

        The moves defining the correction are expected to have their correct number set.

        Args:
            err_move: Move
                the faulty move (has been removed). May be None (if the correction is an "add").
            exp_move: Move
                the correct move (has been added).  May be None (if the correction is a "delete").
        """
        if self.stones_finder is not None:
            self.stones_finder.corrected(err_move, exp_move)

    def confirm_stop(self, process):
        """ A sub-process of this manager that terminates is supposed to pass itself here.
        """
        pass

    def vid_progress(self, progress):
        """ Communicate the progress of the current video read to listeners (GUI, ...).
        """
        pass

    @staticmethod
    def _reflect(name: str, classes: list) -> type:
        """ Import and return the right class based on its name and the available classes definition list.

        If no match found, import and return the class defined in the first tuple of the list.

        Args:
            name: str
                The name of the class to import.
            classes: list of tuples
                Each list element defines a class: the first tuple value is the module, the second is the class name.

        Returns:
            A class object, or
            None if name == "None", or
            None if the first element of the list is (None, None)) and no match is found.

        """
        if name == "None":
            return None
        module_str, class_str = None, None
        for m, c in classes:
            if c == "None":
                continue
            try:
                importlib.import_module(m)
                if module_str is None:
                    module_str, class_str = m, c  # default to the first module that can load
                if c == name:
                    module_str, class_str = m, c  # requested class has been found, stop
                    break
            except ImportError as err:
                print("Can't load {}: {}".format(c, err))
        if class_str not in (None, "None"):
            module = importlib.import_module(module_str)
            return getattr(module, class_str)


class VManager(VManagerBase):
    """ Multi-threaded vision manager.

    Its fields, notably the board and stones finders, must be regarded as (recursively) 'volatile'.
    Concurrency issues are to be expected in this area.

    Attributes:
        daemon: bool
            Inherited from Thread.
        processes: list
            The list of active video processors.
        _interrupt_flag: bool
            True to request the interruption of self.run().
        hasrun: bool
            Indicate if self.run() has completed at least one iteration.
    """

    def __init__(self, controller, imqueue=None, bf=None, sf=None, active=True):
        super().__init__(controller, imqueue=imqueue, bf=bf, sf=sf)
        self.daemon = True
        self.processes = []
        self._interrupt_flag = False
        self.active = active
        self.hasrun = False

    def punch_controller(self):
        super().punch_controller()
        self.controller._pause = self._pause
        self.controller._on = self._on
        self.controller._off = self._off
        self.controller.vidpos = self._set_vidpos
        self.controller.snapshot = self._snapshot

    def _get_capture(self):
        # noinspection PyArgumentList
        cv_capture = get_video_capture(self.controller.video)
        if cv_capture is not None:
            return CaptureReader(cv_capture, self)

    def next(self):
        for proc in self.processes:
            proc.next()

    def run(self):
        """ Initialize the video capture and processors, then periodically listen for change requests.

        Changes may take the form of video input change, or finders class change. The usual reaction is to kill the
        appropriate object and start a fresh one to stay up to date with requests.

        This main loop does not exit by itself, it has to be requested. An exit may also occurs as per the
        'daemon thread' scheme.
        """
        self._register_processes()
        self.init_capt()
        # Main loop, watch for processor class changes and video input changes. Not intended to stop (daemon thread)
        while not self._interrupt_flag:
            if self.active:
                self.check_video()
                if self.capt is not None:
                    self.check_bf()
                    self.check_sf()
            time.sleep(0.2)
            self.hasrun = True

    def interrupt(self):
        self.stop_processing()
        self._interrupt_flag = True

    def stop_processing(self):
        message = "Requesting "
        for proc in self.processes:
            proc.interrupt()
            message += proc.name + ", "
        message = message[0:len(message)-2]
        message += " interruption."
        # release a potential CaptureReader that may keep threads sleeping
        try:
            self.capt.unsync_threads(True)
        except AttributeError:
            pass
        print(message)

    def check_video(self):
        """ Listen for video input source change in controller.

        If the video input source has changed, reset the capture object, then re-instantiate all the finders.
        """
        if self.current_video != self.controller.video:
            self.stop_processing()
            self.init_capt()
            # force a fresh instantiation of each finder
            self.board_finder = None
            self.stones_finder = None
            self.controller.pipe("video_changed")

    def _set_vidpos(self, new_pos):
        """ Set the reading position of the video capture, if it is more than 2% different from current position.

        Ignore small changes to ensure that video progress info pushed by this object is not echoed back by the update
        triggered in the GUI (ugly, but will have to do for now).

        Args:
            new_pos: float
                The video position to set (in % of the total video).

        """
        if self.capt is not None:
            previous = self.capt.get(cv2.CAP_PROP_POS_AVI_RATIO)
            # this method is also triggered by automatic update of the video progress slider when the video is read.
            # ignoring small changes should help prevent risk of annoying rounding effect (could lock the position).
            if 2 < abs(previous * 100 - new_pos):
                self.capt.set(cv2.CAP_PROP_POS_AVI_RATIO, new_pos/100)

    def _snapshot(self, save_goban):
        if self.stones_finder.goban_img is None:
            print("No goban image available to save")
            return
        lastidx = -1
        for previous in [f for f in listdir(snapshot_dir) if os.path.isfile(os.path.join(snapshot_dir, f))]:
            groups = re.findall("(snapshot-)(\d*)(.png)", previous)
            if len(groups) > 0:
                if int(groups[0][1]) > lastidx:
                    lastidx = int(groups[0][1])
        img_name = "snapshot-{}.png".format(lastidx + 1)
        print("Saved " + img_name + " in {}".format(snapshot_dir))
        cv2.imwrite(os.path.join(snapshot_dir, img_name), self.stones_finder.goban_img)
        if save_goban:
            game_name = img_name.replace("snapshot", "game")
            game_name = game_name.replace(".png", ".sgf")
            self.controller.kifu.snapshot(os.path.join(snapshot_dir, game_name))

    def vid_progress(self, progress):
        """ Communicate video read progress to the controller.
        """
        self.controller.pipe("video_progress", progress)

    def check_bf(self):
        """ Listen and react to changes of the requested board finder class.

        If self.board_finder is of different type than self.bf_class, kill the current instance an spawn the right one.
        """
        if type(self.board_finder) != self.bf_class:
            new_name = "None"
            # delete previous instance if any
            if self.board_finder is not None:
                self.board_finder.interrupt()
                while self.board_finder in self.processes:
                    time.sleep(0.1)
                self.board_finder = None  # may help prevent misuse
            # instantiate and start new instance
            if self.bf_class is not None:
                self.board_finder = self.bf_class(self)
                self._spawn(self.board_finder)
                new_name = self.board_finder.__class__.__name__
            self.controller.pipe("select_bf", new_name)

    def check_sf(self):
        """ Listen and react to changes of the requested stones finder class.

        If self.stones_finder is of different type than self.sf_class, kill the current instance an spawn the right one.
        """
        if type(self.stones_finder) != self.sf_class:
            new_name = "None"
            # delete previous instance if any
            if self.stones_finder is not None:
                self.stones_finder.interrupt()
                while self.stones_finder in self.processes:
                    time.sleep(0.1)
                self.stones_finder = None  # may help prevent misuse
            # instantiate and start new instance
            if self.sf_class is not None:
                self.stones_finder = self.sf_class(self)
                self._spawn(self.stones_finder)
                new_name = self.stones_finder.__class__.__name__
            self.controller.pipe("select_sf", new_name)

    def _register_processes(self):
        """ Register "board finders" and "stones finders" with the controller, together with callbacks to start them up.
        """
        for bf_module, bf_class in cvconf.bfinders:
            self.controller.pipe("register_bf", bf_class, self.set_bf_class)
        for sf_module, sf_class in cvconf.sfinders:
            self.controller.pipe("register_sf", sf_class, self.set_sf_class)

    def is_processing(self):
        """ Return True if at least one finder is currently running.
        """
        return len(self.processes).__bool__()  # make the return type clear

    def _on(self):
        """ Turn on video processing if needs be.
        """
        if not self.is_processing():
            # force fresh instantiation (see self.check_bf() for example)
            self.board_finder = None
            self.stones_finder = None
        self.active = True

    def _off(self):
        """ Turn off video processing if needs be.
        """
        if self.is_processing():
            self.stop_processing()
        self.active = False

    def confirm_stop(self, process):
        self.processes.remove(process)
        if process is self.board_finder:
            self.controller.pipe("select_bf", None)
        if process is self.stones_finder:
            self.controller.pipe("select_sf", None)
        print("{0} terminated.".format(process.__class__.__name__))

    def _spawn(self, process):
        """ Start the provided process in a new thread and append it to the list of active processes.

        Args:
            process: VidProcessor
                The process instance to start.
        """
        vt = camkifu.core.video.VisionThread(process)
        process.full_speed = self.full_speed
        self.processes.append(vt)
        # reset a potential CaptureReader to normal behavior: processes should wait for each other when reading frames
        try:
            self.capt.unsync_threads(False)
        except AttributeError:
            pass
        print("{0} starting.".format(process.__class__.__name__))
        vt.start()

    def _pause(self, boolean):
        """ Pause all video processors.
        """
        for process in self.processes:
            process.pause(boolean)

    def set_bf_class(self, bf_class_str):
        """ Set the board finder class attribute of this VManager to match the provided name.

        The expected consequences are to be found in self.check_bf which handles the board finder definition changes.
        """
        bf_class = self._reflect(bf_class_str, cvconf.bfinders)
        if self.bf_class != bf_class:
            self.bf_class = bf_class
        elif not self.is_processing():
            # force restart
            self.board_finder = None
            self.sf_class = None  # a new processing loop has started, forget last choice
            self.stones_finder = None

    def set_sf_class(self, sf_class_str):
        """ Set the stones finder class attribute of this VManager to match the provided name.

        The expected consequences are to be found in self.check_sf which handles the stones finder definition changes.
        """
        sf_class = self._reflect(sf_class_str, cvconf.sfinders)
        if self.sf_class != sf_class:
            self.sf_class = sf_class
        elif not self.is_processing():
            # force restart
            self.stones_finder = None
            self.bf_class = None  # a new processing loop has started, forget last choice
            self.board_finder = None


class CaptureReaderBase:
    """ Wrapper of cv2.VideoCapture to ease the tuning of its usage.

    This base class has an ability to periodically skip frames when reading from a file, aiming at lowering the frame
    rate to mimic a live input and speed up detection. The idea is that there's no need to read 30 frames per second
    when analysing a game of Go, and lowering that figure to 5 frames per second still results in quite a fast sampling
    for this use case. This parameter can be set in cvconf.py.

    Attributes:
        capture: Capture
            Expected to quack like cv2.VideoCapture (object from which actual read() calls will be consumed).
        vmanager: VManager
            The owner of the VidProcessors to synchronize.
        fps: int
            The number of frames per second that should be read from video files (i.e. potentially skip some frames).

    """

    def __init__(self, capture, vmanager: VManagerBase, fps=cvconf.file_fps):
        self.capture = capture
        self.vmanager = vmanager
        self.frame_rate = fps

    def __getattr__(self, item):
        """ Hijack the "read" attribute, delegate all others to self.capture.

        Introduce two extensions points that can be extended to tune frame reading:
            - self.read_file
            - self.downsample
        """
        if item == "read":
            if os.path.isfile(self.vmanager.controller.video):
                return lambda caller: self.downsample(*self.read_file(caller))
            else:
                # no meddling if the input is not a file, yet the argument has to be ignored
                return lambda _: self.downsample(*self.capture.read())
        else:
            return self.capture.__getattribute__(item)

    def read_file(self, caller) -> (bool, object):
        """ Basic file reading extension: skip the necessary number of frames before reading the next one.

        Returns:
            The result of the frame read.

        """
        self.skip()
        return self.capture.read()

    def skip(self) -> None:
        """ Set the next frame number for self.capture, in such a way that self.frame_rate is respected.

        In other word, skip as many frames as needed to match the desired file read frame rate.

        NOTE: on my configuration (mac OS X 10.7, Python 3.4, opencv 3.0.0-beta), when reading a file,
        cv2.VideoCapture.read() seems to regularly skip frames. No more than one at a time, but quite often:
        326 missing frames out of 2904 in my test video (.mov format).
        This phenomenon seems to be deterministic since the skipped frames numbers are always the same.
        See videocapture_skipped_frames() below to test that.

        """
        idx = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        idx += max(1, self.capture.get(cv2.CAP_PROP_FPS) / self.frame_rate)
        idx = min(idx, self.capture.get(cv2.CAP_PROP_FRAME_COUNT))  # don't point after the last frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def downsample(self, ret, img):
        """
        Extension point that can be used to downsample images before returning them to readers.

        """
        # return ret, cv2.pyrDown(img)  # example of reducing each side by a factor of 2
        return ret, img


class CaptureReader(CaptureReaderBase):
    """ VideoCapture wrapper offering synchronization of image consumption for file reading.

    When synchronization is turned on, all VidProcessors that are active must have received the current image before
    the next image is read.

    Attributes:
        buffer:
            The current result from videocapture.read().
        served: set
            The threads that have received the current "buffer" result and should wait for others to have been served.
        lock: Lock
            Used to ensure synchronized calls to self.capture.read().
        sleep_time: float
            The number of seconds the threads that have already been served should sleep to wait for others.
        unsync: bool
            True means that the reading is stopped, and no thread should be kept sleeping.
    """

    def __init__(self, capture, vmanager: VManager):
        super().__init__(capture, vmanager)
        self.buffer = None
        self.served = set()
        self.lock = threading.Lock()
        self.sleep_time = 0.05  # 50 ms
        self.unsync = False

    def read_file(self, caller):
        """ Synchronize file read on all VidProcessor threads, so that they all receive the same sequence of frames.

        Serve the caller immediately if it hasn't received the current frame already, or wait until all active threads
        have been served the current thread before consuming and returning the next frame.

        Note : this method is designed to be called concurrently. Precisely, self.served is supposed to be
        cleared by the last served thread while others are waiting in this method's "sleep" loop.

        Args:
            caller:
                Can be anything that uniquely identifies the calling thread.

        """
        self.init_buffer()
        while not self.unsync and caller in self.served:  # self.served may be cleared by another thread while sleeping
            time.sleep(self.sleep_time)
            self.consume()  # check the possibility that others processors became passive without being noticed
        self.served.add(caller)
        self.consume()
        try:
            return self.buffer[0], self.buffer[1].copy()  # consumers may write on the image, don't expose it.
        except AttributeError:
            return self.buffer

    def init_buffer(self):
        """ Read the very first frame from video capture and buffer it for distribution to all threads.
        """
        with self.lock:
            if self.buffer is None:
                self.buffer = self.capture.read()

    def consume(self):
        """ Consume next image from videocapture (update buffer) if all threads have been served, otherwise pass.
        If unsync has been requested, set buffer to a None result.
        """
        with self.lock:
            if self.unsync:
                self.buffer = False, cvconf.unsynced
                self.served.clear()
            else:
                served_all = True
                for vp in self.active_processes():
                    if vp.processor not in self.served:
                        served_all = False
                        break
                if served_all:
                    self.skip()
                    self.buffer = self.capture.read()
                    self.served.clear()
                    self.vmanager.vid_progress(self.capture.get(cv2.CAP_PROP_POS_AVI_RATIO) * 100)

    def active_processes(self):
        """ Return the list of VidProcessor that are (supposedly) actively reading frames at the moment.
        """
        active = []
        # noinspection PyUnresolvedReferences
        for vidproc in self.vmanager.processes:
            try:
                if vidproc.ready_to_read():
                    active.append(vidproc)
            except AttributeError:
                pass
        return active

    def unsync_threads(self, unsync):
        """ Control the release of all threads waiting on read.

        Args:
            unsync: bool
                True if all threads should be released, False to resume synchronization on file read.
        """
        self.unsync = unsync


class CaptureReaderImg:

    def __init__(self, img):
        if img.endswith('.JPG'):
            print("Warning: '.JPG' format doesn't seem to be supported by OpenCV. Please rename to '.jpg'")
        self.img = cv2.imread(img)
        self.ignored_meths = set()

    def read(self, *args):
        time.sleep(0.5)
        return True, self.img.copy()

    def get(self, *args):
        return 0

    def omega(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        if item not in self.ignored_meths:
            print("CaptureReaderImg: no attribute {}, ignoring".format(item))
            self.ignored_meths.add(item)
        return self.omega


def videocapture_skipped_frames(video="/Path/To/movie.mov"):
    """ (Dev) Analyse the number of frames read by cv2.VideoCapture from the file.
    """
    # noinspection PyArgumentList
    capture = cv2.VideoCapture(video)
    total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("number of frames: {}".format(total))
    ret = True
    prev = 0
    count = 0
    diffs = []
    pos = []
    while ret:
        ret, frame = capture.read()
        x = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if 1 < x - prev:
            diffs.append(x - prev - 1)
            pos.append(x - 1)
        prev = x
        count += 1
    print("total (cv2 property) - observed count = {}".format(total - count))
    print("sum of differences: {}".format(sum(diffs)))
    print("skipped frames:")
    print(pos)
