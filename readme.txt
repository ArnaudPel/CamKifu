
The aim of this software is to record Go games played onto physical boards, by means of video processing. The main drive is to be able to completely focus into the game, and still be able to review it afterwards.
As of 04/01/14, the main value of this software is to provide a testing framework for go-vision algorithms. Hopefully most of the necessary features to quickly throw an idea into code and test it are available.

Current Features (Dec. 2013):
    Automatic detection of the board (+ manual detection available in case of failure)
    Automatic detection of the stones (not robust)
    Ability to select from a list of detection algorithms
    Move, Add, Remove stones manually, even when detection is on
    Save, Open an SGF file. Non-empty sgf can be used as starting point as well

Wish List (decreasing priority):
    Improve the robustness of detection algorithms (†)
    Undo manual goban modification -- the ubiquitous ctrl+Z
    Fast video file processing (currently show-stopping slow)
    Saving metadata about game and players (*)
    Score estimation (*)
    Provide more choice of detection algorithm
    Allow variations. For now only one sequence of moves is recorded
    Navigation improvment -- goto first/last move, ...
    Auto-resize of display

    (†) No matter how many times this is achieved, it is likely to remain the top priority for a while
    (*) In the meantime, the suggestion is to save into a file and use your favorite SGF editor.

    To request a feature, please use the issue tracker.


How to:
Modify the stones on the goban:
    As of Dec. 2013, variations are not supported in a game. This may imply frustrating behavior of the goban interface, compared to already-in-place editors. Hopefully this situation will be forgotten soon. Below are the features currently available.
    Put a stone on the goban:
        - left-click to append a stone of the next color (forbidden when browsing previous moves)
        - hold 'b' or 'w' to insert a move inside a sequence. all subsequent moves will have their number increased by 1.
    Remove a stone:
        - select a stone with left-click, and press 'del'
    Relocate a stone:
        - in order to ease manual correction of stone detection, stones can be dragged across the goban using the usual drag-and-drop interaction (hold left click and move mouse)




Contribution:
--> CL means Command Line

Projects:
    In an attempt to enforce separation of concerns, non-vision Go features are developped as a separate module, named Golib. As of 03/01/14, imports of Golib made in Camkifu are phrased as if both source directories have been merged. Pycharm handles that in Preferences | Project Dependencies. Contributions to improve this module management for non-Pycharm users are welcome.

Code:
    Contribution to any part of the code is obviously welcome at this early stage of the project. Keeping in mind the OCP though, a few class hierarchies are candidates for extension.
        - stonesbase.py and boardbase.py both provide base classes for detection.
        - controller.py and controllerv.py provide levels for MVC interaction.
        - vmanager.py classes can be extended to tune vision threads management.

Testing:
    - camkifutest.py class can be used to provide a reference kifu to check against during processing.
    - To process a sub-part of a video file, see CL argument -b / --bounds.

Guidance:
    Any advice regarding architecture improvements will be received with sharp interest, wether it ends up being put to use or not.



