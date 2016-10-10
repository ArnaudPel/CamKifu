
[Similar projects](http://remi.coulom.free.fr/kifu-snap/)

Below are things to investigate, do, or discard.

## COMPUTER VISION

### Misc

- play with the ton of parameters used locally in cv2 (blurr kernels, iterations, ..., canny thresholds)
- stones finder: adjusting grid with lines detection results seems to lower the number of "crosses" found later by the same lines detection. Re-check this method altogether.
- if the angle between the camera and the board is too large, stones of the front line are often detected one line higher than expected. Investigate more and see to adapt the repartition of the mask ? Some sort of vertical gradient of size or location. The former will imply the introduction of a structure to store all zones areas, at least one per line.
- if board detected at a new location during a game, invalidate all moves found since previous detection (stones buffer)

### Board finder auto

- use a machine learning model. train it either on raw pictures (downsampled), or on a set of lines previously detected with OpenCV canny/hough
- takes forever when `Room 1.mov` is pyrdowned. most likely because bf finds too many lines. investigate
- save background and re-run only if the image is "disturbed" for too long ? Only after several converging detections maybe
- enable manual corrections, which should then disable automatic changes (at least for the corrected corner(s))
- try to make out the vertical side of the goban at the front, it HAS to be excluded
- try to detect black blades around the canonical frame, and relocate accordingly
- create a tester with randomly generated images (lines) to see to what limits bf_auto can be brought to
- try to separate all image pixels in a few color clusters (4-5), and look for one taking the middle of the screen (or 3 for advanced games)

### Contours analysis

- it would be interesting to get an idea, along the contour pixel, where's the interior and where is the exterior, even when contours are not closed. Maybe doable by drawing two contours "enclosing" the initial contour, and computing the length of each. the shortest is interior, and the longest is exterior.
- not closed contours : try to fill them with enclosed circles (stack them in rows and columns)


### OPENCV 3 BETA / Random IDEAS :

- Greatly extended Python bindings, including Python 3 support, and several OpenCV+Python tutorials
- Line Segment Detector (LSD)     -> c++ only (As of Nov 2014)
- Line descriptors and matchers   -> seems to be based on LSD so c++ only as well..


## PYTHON

Please note that a "Fatal Python error: PyEval_RestoreThread: NULL tstate" can occur from time to time and crash the
interpreter itself. It occurs when using the GUI. It seems to be originating in the fact that openCV images have to
be displayed on the main thread, also required by Tkinter. I have dropped the case after several unsuccessful tries,
yet if this problem becomes a pain, try to display fewer images per second.


### General
- create a setup to install Golib in the default "site-packages" location (or redirect from CK !)
- channel all printing to a decent log
- save user preferences: {sgf save dir, ...}.
- extract vision (+ time periods) parameters to file for easier tuning (maybe via GUI)
- try and see if some references could not be [weakened](http://docs.python.org/2/library/weakref.html)
- benchmark.py: parse argument for each run from each reference sgf file ? ex. the first node of the game could contain a custom property where the [--bounds, --moves, --bf, --sf, etc...] arguments could be set
- save a video sample for each game, can be seen as a backup. provide option to disable it (disk space)

### GUI

- add undo command for operations on stones
- add game-related informations (player names, number of captured, time, ... )
- warn when correcting a stone further from its location (e.g. reddening background or circle around stone)
- display "ghost" stones, e.g. for intermediate positions when dragging. or on mouse hover.
- API allowing the display of status lights (to be used by vision).
- API allowing to mark stones with custom colors (to be used by vision).
- create a log viewer pane
--> implement above by extending the GUI from Camkifu in some clever way.

### Go

- implement variation support (proper use of GameTree)
- implement / connect score estimation
