# HOW-TO : VIDEO PROCESSORS

This tutorial is an insight on how the code dealing with video processing has been organized. In one sentence, the `VManager` class is responsible for creating and managing instance(s) of `BoardFinder` and `StonesFinder`, which both extend the `VipProcessor` base class.

## Coordinates systems
A special care has to be observed regarding how matrices indices are used, since `numpy` and `opencv` do not use the same coordinates system. Usually there is as little fiddling with that as can be, but  conversions do occur here and there in the project.
 
If needs be, here's a visual example of the two different logics
 
 ```python
import cv2
import numpy as np

a, b = 20, 100
img = np.zeros((b, b, 3), dtype=np.uint8)
cv2.line(img, (a, 0), (a, b-1), color=(0, 0, 255), thickness=2)  # red opencv line
img[a-1:a+1, :, 1:3] = 255                                       # yellow numpy line
cv2.imshow("Numpy vs OpenCV coord frames", img)
cv2.waitKey()
 ```
 
## On parallelism

Although Python mulit-threading seems not to be parallel in essence because of the [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), thanks to the use of libraries like OpenCV and Numpy, real parallelism is achieved with the 'simple' threading module. Therefore I didn't feel it necessary to use the multiprocessing module for this project. Any heads-up on this matter is welcome, as always.
 
## Goodies

### Metadata display

A `VidProcessor` has the ability to accumulate strings (lines), and draw them on the next image passed to `_show()`. The positionning of the lines is handled automatically, starting at the bottom left of the image. The data is cleared right after `self._show()` has been called, so it has to be refilled for each image to be shown.

All there is to do is fill `metadata`, wich is a [`defaultdict(list)`](https://docs.python.org/3/library/collections.html#collections.defaultdict) with values mapped to `str` keys representing the text. The keys must indicate where to insert the value with the `{}` marker, since [`str.format`](https://docs.python.org/3/library/stdtypes.html#str.format) will be called.

```python
self.metadata["a line {}"] = (42, -42)
```

Ok but why use a defaultdict ? Because it provides the ability to quickly append multiples values for the same key, which can be very handy to accumulate data over successive unshown frames (again, showing a frame will clear all the keys and values).

```python
self.metadata["another line : {}"].append(19)
self.metadata["another line : {}"].append("nineteen")
```
