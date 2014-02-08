
Camkifu
=======

**This project depends on Golib. See below.**

The aim of this software is to record Go games played on physical boards, by means of video processing. The main drive is to be able to focus on a game, and be able to review it afterwards.

As of 04/01/14, the main value of this software is to provide a testing framework for go-vision algorithms. Hopefully most of the necessary features to quickly throw an goban image processing idea into code and test it are available.

I have started a new job recently, so my devs will stop for an undetermined time. As long as this sentence is displayed on Github, I guaranty a response to any question about this project.

Current Features (Dec. 2013)
----------------------------
    Automatic detection of the board (+ manual detection available in case of failure)
    Automatic detection of the stones (not robust)
    Ability to select from a list of detection algorithms
    Move, Add, Remove stones manually, even when detection is on
    Save, Open an SGF file. Non-empty sgf can be used as starting point as well
    Goto a specified move `(ctrl+G)`

Wish List (decreasing priority)
-------------------------------
    Improve the robustness of detection algorithms (†)
    Support undo of manual goban modifications -- the ubiquitous `ctrl+Z`
    Fast video file processing (currently show-stopping slow)
    Saving metadata about game and players (*)
    Score estimation (*)
    Provide more choice of detection algorithm
    Allow variations. For now only one sequence of moves is recorded
    Auto-resize of goban display

    (†) This is likely to remain the top priority for a while
    (*) In the meantime, the suggestion is to save into a file and use your favorite SGF editor.


Contribution Note
-----------------

### Projects
    In an attempt to enforce separation of concerns, non-vision Go features are developped as a separate module, named Golib. As of 03/01/14, imports of Golib made in Camkifu are phrased as if both source directories have been merged. Pycharm handles that in Preferences | Project Dependencies. Contributions / advice to improve the module management for non-Pycharm users are welcome.

### Code
    Keeping in mind the OCP, a few class hierarchies are favorite candidates for extension.
        - `stonesbase.py` and `boardbase.py` both provide base classes for detection.
        - `controller.py` and `controllerv.py` provide levels for MVC interaction (notice the "v" difference in filename).
        - `vmanager.py` classes can be extended to tune vision threads management (life and interactions).

### Testing
    - `camkifutest.py` class can be used to provide a reference kifu during processing, that can record errors.
    - testing options (like processing a subpart of a file, checking against a subpart of a reference sgf) can be found in `camkifutest.py`, in the form of Command Line arguments.
    - possibility to save the last location of the board for `BoardFinderManual`, look for `.savez(gobanloc_npz)` as commented code of that class. As of 14/01/14, it is in the `onmouse()` method.

### Guidance
    Any advice regarding architecture improvement will be received with sharp interest, wether it ends up being put to use or not.



