# HOW-TO : VIDEO PROCESSORS

This tutorial is an insight on how the code dealing with video processing has been organized. In one sentence, the `VManager` class is responsible for creating and managing instance(s) of `BoardFinder` and `StonesFinder`, which both extend the `VipProcessor` base class.

## On parallelism

Although Python multi-threading code is not parallel in essence because of the [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), thanks to the use of libraries like OpenCV and Numpy, real parallelism is achieved with the 'simple' multi-threading module. Therefore I didn't deem the multiprocessing module necessary for this project. Any critic on this matter is welcome, as always.


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

- drawing each intersection zone (linear then border)
- suggesting a move to the kifu (pretending we have successfully detected a stone)
- drawing values on the grid
- updating the grid location
- using a mask to isolate each intersection in a circle (will it stay ???)

Alright let's see.

#### Intersections: zone

The base method `_getzone` returns, for a goban's lines intersection, the corresponding pixel zone. Let's get a feeling by displaying the 37 zones representing the diagonals of the goban:
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r in range(gsize):      # row index
        for c in range(gsize):  # column index
            if r == c or r == gsize - c - 1:
                zone, pt = self._getzone(goban_img, r, c)
                canvas[pt[0]:pt[2], pt[1]:pt[3], :] = zone
    self._show(canvas)
```

`zone` is a numpy array containing the pixels of the required subregion of the image. `pt`, for "point", gives the coordinates of that (rectangle) zone inside the image: `pt = [x_start, y_start, x_end, y_end]` 

Note that by default, the values returned by `_getzone` are merely calculated by dividing the image in 19 by 19 (`gsize * gsize`) zones without any analysis. As explained above, the default (most simple) approach places entire trust in the board finding feature.

#### Intersections: iteration

We can now introduce `_empties_border` and `_empties_spiral` generators. They enumerate intersections where no stone has been recorded (either by vision or the user) to be present on the goban. These intersections are obviously privileged locations to analyse when looking for new stones. Let's update our iteration code to have a look.
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r, c in self._empties_border(2):  # 2 is the line height as in go vocabulary (0-based)
        zone, pt = self._getzone(goban_img, r, c)
        canvas[pt[0]:pt[2], pt[1]:pt[3], :] = zone
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
            zone, pt = self._getzone(goban_img, r, c)
            self.canvas[pt[0]:pt[2], pt[1]:pt[3], :] = zone
            break
        count += 1
    self.last_shown = 0  # force display of all images
    self._show(self.canvas)
```

Alright.