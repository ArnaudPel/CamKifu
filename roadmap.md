
Duality arises from Unity, -- But do not be attached to this Unity

to enable automatic ssh server startup on Jack :
sudo mv /etc/init/ssh.conf.disabled /etc/init/ssh.conf

links to similar projects : http://remi.coulom.free.fr/kifu-snap/


## VISION
- board finder auto : takes for ever when image from Room 1.mov is pyrdowned. most likely because it finds too many lines. investigate
- board finder auto : save background and re-run only if the image is "disturbed" for too long ? Only after several converging detections maybe
- board finder auto : enable manual corrections, which should then disable automatic changes (at least for the corrected corner(s))
- board finder auto : try to make out the vertical side of the goban in front, it HAS to be excluded
- board finder auto : try to detect black blades around the canonical frame, and relocate accordingly.
- board finder auto : create a tester with randomly generated images (lines) to see to what limits it can be brought to
- board finder auto : try to separate image in a few clusters (4-5), and look for one taking the middle of the screen (or 3 taking the middle for advanced games)


## BEFORE PYTHON PUBLISH:

- make more videos, in different conditions. especially play on the first line, to test limit conditions.

- CODE REVIEW
    * check bugs file
    * go through todos, do them or move to this file so they get out of the way
    * documentation
        - first read carefully [jetbrains' heads-up](https://www.jetbrains.com/pycharm/webhelp/documenting-source-code-in-pycharm.html), [pep3107](https://www.python.org/dev/peps/pep-3107/)
        - check and fix existing doc
        - create missing doc
        - check and fix comments
        - add messages to assertions  `assert cond() , "message"`
- create a setup to install Golib in the default "site-packages" location (or redirect from CK !)
- license
- clean up and mark down this file :)


## ON PYTHON PUBLISH:

- indicate that some bugs may be due to the moving from (python, opencv)2 to 3
- tutos on how to extend the code (minial board and stones finder, how to communicate to the GUI)
- links to openCV install
- links to PyCharm CE
- jetbrains has a banner generator. can we set one on github ?
- on contour analysis:
    * it would be interesting to get an idea, along the contour pixel, where's the interior and where is the exterior, even when contours are not closed. Maybe doable by drawing two contours "enclosing" the initial contour, and computing the length of each. the shortest is interior, and the longest is exterior.
    * not closed contours : try to fill them with enclosed circles (stack them in rows and columns)


## OPENCV 3 BETA / Random IDEAS :

- Greatly extended Python bindings, including Python 3 support, and several OpenCV+Python tutorials
- Line Segment Detector (LSD)     -> c++ only (As of 24/10/14)
- Line descriptors and matchers   -> seems to be based on LSD so c++ only as well..


## BEFORE CPP PUBLISH:

- try out things that were not available in python (startFindContours, freeman chaincode for contours, )
- create a boardfinder_test.py that can compare detection against known locations.
- point ck_test to "test/res/ck" and make CLI args optional (process all input in that folder)
- write more test (+ code coverage tool ?), + provide videos and sgf ressources
- try the VidRecorder again (as of opencv-2.4.8)
- make vision values as dynamic as possible (frame size, ...)


## ON CPP PUBLISH:

- links to openCV install
- set issue tracker
- convert this file into milestones
- add myself to:
    * http://stackoverflow.com/q/5742140/777285  (+ crawl from there)
    * kgs and igs profiles
- hammer keywords online (go, weiqi, baduk, igo, camera, recording, video) to be found


## CODE ROADMAP FOR COMMUNITY :

Please note that a "Fatal Python error: PyEval_RestoreThread: NULL tstate" can occur from time to time and crash the
interpreter itself. It occurs when using the GUI. It seems to be originating in the fact that openCV images have to
be displayed on the main thread, also required by Tkinter. I have dropped the case after several unsuccessful tries,
yet if this problem becomes a pain, try to display fewer images per second and it should soothe.


### PYTHON

- channel all printing to a decent log
- create a log viewer pane in the gui
- save user preferences: {sgf save dir, ...}.
- extract vision (+ time periods) parameters to file for easier tuning (maybe via GUI)
- try and see if some references could not be weakened  (http://docs.python.org/2/library/weakref.html)

### GUI

- add undo command (e.g. moving or deleting a stone)
- add game-related informations (player names, number of captured, time, ... )
- warn when correcting a stone further from its location (e.g. reddening background or circle around stone)
- display "ghost" stones, e.g. for intermediate positions when dragging. or on mouse hover.
- auto resize of goban and stones according to window size.
- API allowing the display of status lights (to be used by vision).
- API allowing to mark stones with custom colors (to be used by vision).
--> implement above by extending the GUI from Camkifu in some clever way.

### GO

- implement variation support (proper use of GameTree)
- implement / connect score estimation

### CV

- allow initial game setup auto detection, like handicap stones
- provide vidprocessors with their last successful run (interruption awareness)
- save a video sample for each game, can be seen as a backup. provide option to disable it (disk space)
- update background periodically
- when board detected at a new location, invalidate all moves found since previous detection (stones buffer)
- bg-sub: update B/W thresholds dynamically (per zone ?). Increase on detect, lower on user stone add.
- ability to provide image file as input (for testing)
- ability to naviguate movie file, based on already detected moves. Eg test has passed but for a few elems, provide a way to jump before them yet as close as possible.

---------------------------------------------------------------------------------------------------

