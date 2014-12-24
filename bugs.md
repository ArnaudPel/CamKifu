


## BUGS

image hidding (image not shown) doesn't seem to work as expected when one vidprocessor owns 2 images.

stones finder bulk update may suggest moves impossible in the game of Go. So the rules object will refuse and crash. things don't seem to go back to normal aftewards, so check that the game's state is not corrupted.

check the drag feature, if the mouse goes too far there may be exceptions risen.


## WON'T FIX
- a "Fatal Python error: PyEval_RestoreThread: NULL tstate" can occur. I believe this is due to the fact that openCV cv2.show() must be run on the main thread, together with Tkinter in this case. A workaround to that, on my machine at least, was to force the reduction of images display frequency. This bug is most likely dependant on available CPU, so the display frequency can probably be brought up on fast machines.

- when clicking on an already running process in the menu, it restarts it. seems a bit extreme, maybe do nothing -- won't fix for now

- the save changes dialog doesn't show up when hitting CMD+Q  -- won't fix : this seems to be a pain with Tkinter, no time for that.
