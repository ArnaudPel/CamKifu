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

## Stones detection

This part aims at demonstrating how to create and integrate a new stones detection algorithm frorm scratch. After the minimal version has been put together, a tour of util functions around stones finding is offered.

### 1. Minimal

The base class `StonesFinder` provides a default implementation of the workflow of a stones finding routine. By workflow, read the periodic analysis of goban frames in order to dectect the add of new stones. Two abstract methods require our attention.

 ```python
class StonesFinderTuto(StonesFinder):
        
    def _find(self, goban_img):
        pass

    def _learn(self):
        pass
 ```
 
The `_find` method is where the main detection algorithm is supposed to be implemented. This method will be called periodically by the base class, provided each time with a fresh frame. Thanks to the (hopefully) good work of the board finder, `goban_img` can be expected to contain the goban pixels only, with a perspective transform already applied. In other words, it can be expected to represent a picture of the goban from above, taking the entire image.

The `_learn` method is called when a user has made a manual correction on the kifu : added, removed or relocated a stone. We'll go into more details about this later.
 
Ok. In order to visualize our minimal implementation success, let's add a couple of lines.

```python
def _find(self, goban_img):
        draw_str(goban_img, "Hello stones finding tutorial !")
        self._show(goban_img)
```

But how is Camkifu supposed to know about our new class ? We tell it by adding the class to `cvconf.py`, together with the other stones finding guys already listed:

```python
from camkifu.stone.sf_tuto import StonesFinderTuto
...
# the first element in the list will be loaded at startup.
sfinders = [
    StonesFinderTuto,
    ...
]
```

Alright, it should be good to go ! Oups, no, forgot the label, which provides the display name of our finder for the GUI. So here is the complete minimal implementation.

```python
from camkifu.core.imgutil import draw_str
from camkifu.stone.stonesfinder import StonesFinder


class StonesFinderTuto(StonesFinder):

    label = "Stones Tuto"

    def _find(self, goban_img):
        draw_str(goban_img, "Hello stones finding tutorial !")
        self._show(goban_img)

    def _learn(self):
        pass
```

### 2. Algorithm util

Now that we've integrated into the flow, time to do some work. This part will focus on the util methods provided by the `StonesFinder` base class.

- drawing values on the grid
- updating the grid location
- using a mask to isolate each intersection in a circle (will it stay ???)

Alright let's see.

#### Move suggestion

When a move (new stone played) has been detected by vision, this information has to be communicated. The `suggest` method helps with that:

```python
from golib.config.golib_conf import B, W

def _find(self, goban_img):
    # check emptiness to avoid complaints since this method will be called in a loop
    if self.is_empty(2, 12):
        # using "numpy" coordinates frame for x and y
        self.suggest(B, 2, 12)
    if self.is_empty(12, 2):
        # using "opencv" coordinates frame for x and y
        self.suggest(B, 2, 12, 'tk')
```

This example also makes for a good reminder of the kind of confusion that can occur between numpy and opencv coordinates frames.

#### Intersections: zone

The base method `_getrect` returns, for a goban's lines intersection, the corresponding pixel zone. Let's get a feeling by displaying the 37 zones representing the diagonals of the goban:
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r in range(gsize):      # row index
        for c in range(gsize):  # column index
            if r == c or r == gsize - c - 1:
                x0, y0, x1, y1 = self._getrect(r, c)
                canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
    self._show(canvas)
```

`zone` is a numpy array containing the pixels of the required subregion of the image. `pt`, for "point", gives the coordinates of that (rectangle) zone inside the image: `pt = [x_start, y_start, x_end, y_end]` 

Note that by default, the values returned by `_getrect` are merely calculated by dividing the image in 19 by 19 (`gsize * gsize`) zones without any analysis. As explained above, the default (most simple) approach places entire trust in the board finding feature.

### 3. Other misc. goodies

Things that had seemed to make sense at some point, but may not anymore.

#### Intersections: iteration

Let's introduce the `_empties_border` and `_empties_spiral` generators. Their job is to enumerate intersections where no stone has been recorded (either by vision or the user). These intersections are privileged locations to analyse when looking for new stones. Let's have a look.
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r, c in self._empties_border(2):  # 2 is the line height as in go vocabulary (0-based)
        x0, y0, x1, y1 = self._getrect(r, c)
        canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
    self._show(canvas)
```

So basically `_empties_border` yields positions along a ring situated at the provided height (0 being the outermost ring). And `empties_spiral` is just calling `_empties_border` iteratively from the outermost ring to the center. Let's use the total frame count and an accumulated image to see the spiral effect.

```python
def __init__(self, vmanager):
    super().__init__(vmanager)
    self.canvas = None

def _find(self, goban_img):
    count = 0
    if self.canvas is None:
        self.canvas = zeros_like(goban_img)
    for r, c in self._empties_spiral():
        if count == self.total_f_processed % gsize**2:
            x0, y0, x1, y1 = self._getrect(r, c)
            self.canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
            break
        count += 1
    self.last_shown = 0  # force display of all images
    self._show(self.canvas)
```

Alright.


## On parallelism

Although Python mulit-threading seems not to be parallel in essence because of the [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), thanks to the use of libraries like OpenCV and Numpy, real parallelism is achieved with the 'simple' threading module. Therefore I didn't feel it necessary to use the multiprocessing module for this project. Any heads-up on this matter is welcome, as always.