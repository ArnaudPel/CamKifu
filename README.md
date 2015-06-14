
# Camkifu

The aim of this software is to record Go games played on physical boards, by means of video processing. The main drive is to be able to focus fully on the game, while being able to review and save it.

As of June 2015, the main value of this software is to provide a testing framework for go-vision algorithms. Hopefully most of the necessary features to quickly throw a goban image processing idea into code and test it are available. Find more in **Current Features** and **Contribution note** below.

**This project depends on:**

- Golib. See below
- [OpenCV 3](http://opencv.org/)
- [Numpy 1.9](http://www.numpy.org/)
- [Python 3](https://www.python.org/downloads/)

## Current Features (June 2015)
- Automatic detection of the board (+ manual detection available in case of failure)
- Automatic detection of the stones (not robust)
- Ability to select from a list of detection algorithms from the GUI
- Move, Add, Remove stones manually, whether detection is on or not
- Automatic pause of vision processing when browsing moves (variations are currently not supported)
- Save, Open an SGF file. Non-empty sgf can be used as starting point as well

## Wish List
By decreasing priority

1. Improve the robustness of detection algorithms (†)
2. Support undo of manual goban modifications -- the ubiquitous `ctrl+Z`
3. Fast video file processing
4. Saving metadata about game and players (*)
5. Score estimation (*)
6. Provide more choice of detection algorithm
7. Allow variations (branching) in moves sequence. For now only one single line of play is supported.
8. Auto-resize of goban display

(†) This is likely to remain the top priority for a while
(*) In the meantime, the suggestion is to save into a file and use your favorite SGF editor.


## Contribution Note

### Projects
In an attempt to enforce separation of concerns, non-vision Go features are developped as a separate module, named Golib. As of June 2015, imports of Golib made in Camkifu are phrased as if both source directories have been merged. [Pycharm](http://www.jetbrains.com/pycharm/) handles that in `Preferences | Project Dependencies`. Contributions / advice to improve the module management are welcome.

### Code
Keeping in mind the OCP, a few class hierarchies are favorite candidates for extension.

- `stonesfinder.py` and `boardfinder.py` both provide base classes for detection. There already are example implementations partially meeting the functional expectations (they still fall short of robustness).
- `controller.py` and `controllerv.py` provide levels for MVC interaction (notice the "v" difference in filename).
- `vmanager.py` classes can be extended/modified to tune vision threads management (life and interactions).

Some work has been done around board/stones finders multi-thread flow:

- File read synchronization is enabled by default, meaning that concurrent video processors are seeing the same image at the same time when reading input from a file. This is paramount if the board finding process is run in parallel of stones finding, as one finder may be several times faster than the other(s). There is no such concern when processing live video feed, since all finders can read a new image and be synchronized on the "present moment".
- Disabling of features so long as other threads don't provide the necessary data. Very basically, disable stones finding as long as the board location is not known.

### Testing
- `DetectionTest.py` can be used to compare algorithms against known expected results, and calculate the match percentage.
- testing options (like processing a subpart of a file, checking against a subpart of a reference sgf) can be found in `DetectionTest.py`, in the form of Command Line arguments.


## (Metaphysical) Thanks

I want to acknowledge the fact that this project has little to do with me. It is the work of millions of people.

- From the food I eat to the computer I've been using to learn and code all those years. A big global thanks and compationate thought to the people who have / are suffering in the world so that some of us may enjoy the grace of sitting in front of a computer most of the day.
- Focusing now on the developer community, thanks to all the designers, programmers, prophets that have made so many systems, languages and great libraries available to us. Python contributors, OpenCV contributors, Stackoverflow, thanks a lot.
- Finaly, I can't help but word a special thanks to the Jetbrains team, who built such awesome friends as Pycharm and Intellij Idea. I often wonder at what a proper job they've been doing so far at making coding an even more thrilling thing than it is by essence. Love you guys.
 
 Cheers