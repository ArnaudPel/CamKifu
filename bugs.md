


## BUGS

all clear


## WON'T FIX
- a "Fatal Python error: PyEval_RestoreThread: NULL tstate" can occur. I believe this is due to the fact that openCV cv2.show() must be run on the main thread, together with Tkinter in this case. A workaround to that, on my machine at least, was to force the reduction of images display frequency. This bug is most likely dependant on available CPU, so the display frequency can probably be brought up on fast machines.

- the save changes dialog doesn't show up when hitting CMD+Q  -- won't fix : this seems to be a pain with Tkinter, no time for that.
