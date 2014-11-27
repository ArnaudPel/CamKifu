
Duality arises from Unity, -- But do not be attached to this Unity

to enable automatic ssh server startup on Jack :
sudo mv /etc/init/ssh.conf.disabled /etc/init/ssh.conf

links to similar projects : http://remi.coulom.free.fr/kifu-snap/



## BEFORE PYTHON PUBLISH:

- stones finder behavior:
    * start by trying to take a kmeans picture (with pertinence check) in case we don't start from scratch
    * in low stones density regions, look for stones with contours. as soon as possible, switch to kmeans (do it per subzone)
    * much need for analysis of contours that are not closed. look for arc-of-circle detection ? or google the problem itself
    * discard frames / wait for changes using background analysis
- make more videos, in different conditions
- add left-click listener on goban (golib) and provide menu to invert stone color (plus repeat existing commands maybe)
- have an automatic downsampling before processing(s) ? in boardfinder at least (since the canonical frame size is fixed already).
- try to have some more fun w board detection and stones detection to show a nice face to the world
    * Try a 2means board detector ? Maybe coupled with edges detection in a second stage.
    * Smart segmentation and edge-aware filters ?
    * Page 267 and on for background detection modeling
- go through todos
- fix in-code documentation  --> especially check existing doc  --> add messages to assertions ! (assert cond() , "message")
- shoot as many pycharm warnings as possible
- create a setup to install Golib in the default "site-packages" location (or redirect from CK !)
- fix author (replace any Kohistan with A.P.)
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

