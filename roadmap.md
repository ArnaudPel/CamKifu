
Duality arises from Unity, -- But do not be attached to this Unity

to enable automatic ssh server startup on Jack :
sudo mv /etc/init/ssh.conf.disabled /etc/init/ssh.conf

links to similar projects : http://remi.coulom.free.fr/kifu-snap/



## BEFORE PYTHON PUBLISH:

- make more videos, in different conditions
- add a "Next" button for developers to atomically control frame read, associated with a self.wait_next() in VidProcessor  AND use that occasion to record a tuto on how to add something to the GUI.
- how about merging ControllerV and VManager to reduce serving hatches ? (passe-plat)
- have an automatic downsampling before processing(s) ? in boardfinder at least (since the canonical frame size is fixed already).
- try to have some more fun w board detection and stones detection to show a nice face to the world
    * Smart segmentation and edge-aware filters ?
    * Page 267 and on for background detection modeling
- go through todos
- fix in-code documentation  --> especially check existing doc
- shoot as many pycharm warnings as possible
- create a setup to install Golib in the default "site-packages" location (or redirect from CK !)
- license


## ON PYTHON PUBLISH:

- indicate that some bugs may be due to the moving from (python, opencv)2 to 3
- tutos on how to extend the code (minial board and stones finder, how to communicate to the GUI)
- links to openCV install
- links to PyCharm CE
- jetbrains has a banner generator. can we set one on github ?


## OPENCV 3 BETA :

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
- clean up and mark down this file :)


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
- API allowing to mark stones with colors (to be used by vision).
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
