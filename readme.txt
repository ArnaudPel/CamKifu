
The aim of this software is to record Go games played onto physical boards, by means of video processing. The main drive is to be able to completely focus into the game, and still be able to review it afterwards.

Current Features (Dec. 2013):
    Automatic detection of the board (+ manual detection available in case of failure)
    Automatic detection of the stones
    Move, Add, Remove stones anytime, with or without detection running.
    Save, Open an SGF file. Non-empty sgf can be used to restart a postponed game

Wish List (decreasing priority):
    Improve the robustness of detection algorithms (†)
    Undo manual goban modification -- the ubiquitous ctrl+Z
    Transparent file / live video input
    Saving metadata about game and players (*)
    Score estimation (*)
    Choice of different Detection algorithms
    Allow variations. For now only one sequence of moves is recorded
    Navigation improvment -- goto first/last move, ...
    Auto-resize of display

    (†) No matter how many times this is achieved, it is likely to remain the top priority for a while
    (*) In the meantime, the suggestion is to save into a file and use your favorite SGF editor.

    To request a feature, please use the issue tracker.


How to:
Modify the stones on the goban:
    As of Dec. 2013, variations are not supported in a game. This may imply frustrating behavior of the goban, compared to already-in-place editors. Hopefully this situation will be forgotten soon. Below are the current features available.
    Put a stone on the goban:
        - left-click to append a stone to game. This can only be done when the goban is displaying the last stone of the game
        -





Contribution:
Code:
    Keeping in mind the OCP, a few class hierarchies have been put to good use. Hopefully they will provide extension points for further developments.

Guidance:
    Any advice regarding architecture improvements will be received with sharp interest, wether it ends up being put to use or not.