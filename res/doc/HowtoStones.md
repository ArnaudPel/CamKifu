# HOW-TO : STONES FINDER
Before reading further, you may be intersted in having a look at `Howto VidProcessor.md`.

This part aims at demonstrating how to create and integrate a new stones detection algorithm frorm scratch. After the minimal version has been put together, a tour of util functions around stones finding is offered.

## 1. Minimal

The abstract base class `StonesFinder` provides a default implementation of the workflow of a stones finding routine. By workflow, read the periodic analysis of goban frames in order to dectect the add of new stones. The minimal implementation must overwrite the following two abstract methods:

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

Alright, it should be good to go ! So here is the complete minimal implementation.

```python
from camkifu.core.imgutil import draw_str
from camkifu.stone.stonesfinder import StonesFinder

class StonesFinderTuto(StonesFinder):

    def _find(self, goban_img):
        draw_str(goban_img, "Hello stones finding tutorial !")
        self._show(goban_img)

    def _learn(self):
        pass
```

## 2. Algorithm util

Now that we've integrated into the flow, time to do some work. This part will focus on the util methods provided by the `StonesFinder` base class.

- drawing values on the grid
- updating the grid location
- using a mask to isolate each intersection in a circle (will it stay ???)
- constraint checking attempts (pros and cons)

Alright let's see.

### Move suggestion

When a move (new stone played) has been detected by vision, this information has to be communicated. The `suggest` method helps with that:

```python
from golib.config.golib_conf import B, W

def _find(self, goban_img):
    # check emptiness to avoid complaints since this method will be called in a loop
    if self.is_empty(2, 12):
        # using "numpy" coordinates frame for x and y
        self.suggest(B, 2, 12)
```

In case of multiple detection, the `bulk_update` method helps grouping the moves (appends or deletions) into a single update, enabling performance gain since the GUI is only updated when needs be.

```python
from time import sleep
from golib.config.golib_conf import gsize, B, W, E

def _find(self, goban_img):
        # using "numpy" coordinates frame for x and y
        black = ((W, 8, 8), (W, 8, 10), (W, 10, 8), (W, 10, 10))
        white = ((B, 7, 7), (B, 7, 11), (B, 11, 7), (B, 11, 11), (B, 9, 9))
        add = black if self.total_f_processed % 2 else white
        rem = white if self.total_f_processed % 2 else black
        moves = []
        for color, r, c in add:
            moves.append((color, r, c))
        for _, r, c in rem:
            if not self.is_empty(r, c):
                moves.append((E, r, c))  # indicate removal with the 'E' (empty) color
        sleep(0.7)
        
        self.bulk_update(moves)
```


### Intersections: zone

The base method `getrect` returns, for a goban's lines intersection, the corresponding pixel zone. Let's get a feeling by displaying the 37 zones representing the diagonals of the goban:
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r in range(gsize):      # row index
        for c in range(gsize):  # column index
            if r == c or r == gsize - c - 1:
                x0, y0, x1, y1 = self.getrect(r, c)
                canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
    self._show(canvas)
```

`zone` is a numpy array containing the pixels of the required subregion of the image. `pt`, for "point", gives the coordinates of that (rectangle) zone inside the image: `pt = [x_start, y_start, x_end, y_end]` 

Note that by default, the values returned by `getrect` are merely calculated by dividing the image in 19 by 19 (`gsize * gsize`) zones without any analysis. As explained above, the default (most simple) approach places entire trust in the board finding feature.

## 3. Other misc. goodies

Things that had seemed to make sense at some point, but may not anymore.

### Intersections: iteration

Let's introduce the `_empties_border` and `_empties_spiral` generators. Their job is to enumerate intersections where no stone has been recorded (either by vision or the user). These intersections are privileged locations to analyse when looking for new stones. Let's have a look.
 
```python
from numpy import zeros_like
from golib.config.golib_conf import gsize

def _find(self, goban_img):
    canvas = zeros_like(goban_img)
    for r, c in self._empties_border(2):  # 2 is the line height as in go vocabulary (0-based)
        x0, y0, x1, y1 = self.getrect(r, c)
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
            x0, y0, x1, y1 = self.getrect(r, c)
            self.canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
            break
        count += 1
    self.last_shown = 0  # force display of all images
    self._show(self.canvas)
```

That's it for this tuto !
