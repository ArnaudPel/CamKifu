# HOW-TO : RUN CONFIGURATIONS

This tutorial aims at presenting the different runnable scripts available. All the usage information provided below can be obtained by using the `--help` command line argument

## Golib

The most basic runnable is the GUI of project Golib.

`Golib/src/glmain.py`
```
usage: glmain.py [-h] [--sgf SGF]

optional arguments:
  -h, --help  show this help message and exit
  --sgf SGF   SGF file to load at startup.
```

## Camkifu

The default "vision" runnable is in Camkifu. The current implementation is multi-threaded (as per `VManager`). 

`CamKifu/src/ckmain.py`
```
usage: ckmain.py [-h] [--sgf SGF] [-v VID] [-b R R] [--bf BF] [--sf SF]

optional arguments:
  -h, --help            show this help message and exit
  --sgf SGF             SGF file to load at startup.
  -v VID, --video VID   Filename, or device, as used in cv2.VideoCapture(). Defaults to device "0".
  -b R R, --bounds R R  Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()
  --bf BF               Board finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
  --sf SF               Stones finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
```

## Camkifu - dev

When suspecting a concurrency issue in the code, the ability to run Camkifu in a "single threaded" mode can be handy.

`CamKifu/test/mains/SingleThreaded.py`

```
usage: see ckmain.py
```
This configuration can also come in handy when in need to sequentially display subregions of the current image with `cv2.waitKey()`. This is not possible in the "multi-threaded" environment because, as of opencv 3.0.0-beta, `cv2.imshow()` can only be called on the main thread, while the vision algorithms run on their own thread. There is support for images display from vision threads by use of a `Queue` that passes them to the main thread, but it's not designed to offer control via `cv2.waitKey()`.

## Camkifu - test

When having put together a nice algorithm, comes the need to see how well it performs against a benchmark of videos. The following configuration can help with the automation of checking results from one video against its expected move sequeneces (sgf files).

`CamKifu/test/mains/detectiontest.py`

```python
# TODO usage
```
