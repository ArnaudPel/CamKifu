

This is a file I was using when coding on my own. Should be discarded for a more proper issue tracking.

## OPEN BUGS

-


## WON'T FIX

### Fatal Python error: PyEval_RestoreThread: NULL tstate
Has been occasionnally and randomly occuring for quite a while. I believe this is due to the fact that `cv2.show()` must be run on the main thread, together with `Tkinter` in this case. A workaround for that, on my machine at least, was to force the reduction of images display frequency. This bug is most likely dependant on available CPU, so the display frequency can probably be raised up on fast machines.

### Fatal Python error: GC object already tracked
Has started to occur randomly of late. I suspect it is coming from some concurrency issue brought by recent changes in `SfMeta`. Or it could originate from the way `VManager` is reading frames. Since it may also be on the opencv side -currently playing with opencv 3.0.0-beta-, or dependent on my mac OS X build, I won't spend time trying to fix it now. [ref1](http://pyrit.wordpress.com/2010/02/18/385/), [ref2](http://stackoverflow.com/questions/23178606/debugging-python-fatal-error-gc-object-already-tracked).

### Misc
- The save changes dialog doesn't show up when hitting `CMD`+`Q`. This question seems to be a pain with Tkinter, no time for that.
