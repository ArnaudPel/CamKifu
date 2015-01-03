# HOW-TO : RUN CONFIGURATIONS

This tutorial aims at presenting the different runnable scripts available. Most of the usage information provided below can be obtained by using the `--help` command line argument

## Golib

The most basic runnable is the GUI of project Golib.

`Golib/src/glmain.py`
```
usage: glmain.py [-h] [--sgf SGF]

optional arguments:
  -h, --help  show this help message and exit
  --sgf SGF   SGF file to load at startup.
```

Example:
```
$ python3 /Path/To/glmain.py --sgf /Path/To/mygame.sgf
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

Examples:
```
$ python3 /Path/To/ckmain.py -v /Path/To/video.avi
$ python3 /Path/To/ckmain.py --sgf /Path/To/mygame.sgf -v /Path/To/video.avi -b 0.1 0.9  --bf BoardFinderAuto --sf SfContours
```

## Camkifu - dev

When suspecting a concurrency issue in the code, the ability to run Camkifu in a "single threaded" mode can be handy.

`CamKifu/test/mains/SingleThreaded.py`

```
usage and example: see ckmain.py
```
This configuration is also vital when in need to sequentially display subregions of the current image with `cv2.waitKey()`. This is not possible in the "multi-threaded" environment because, as of opencv 3.0.0-beta, `cv2.imshow()` can only be called on the main thread, while the vision algorithms run on their own thread. There is support for images display from vision threads by use of a `Queue` that passes them to the main thread, but it's not designed to offer fine control via `cv2.waitKey()`.

## Camkifu - test

When having put together a nice algorithm, comes the need to see how well it performs against a few known videos. The following configuration can help with the automation of checking results from one video against its expected moves sequence (an sgf file).

`CamKifu/test/mains/detectiontest.py`

```
usage: detectiontest.py --sgf SGF [-h] [-v VID] [-b R R] [--bf BF] [--sf SF] [--failfast] [-m M M]

required argument:
  --sgf SGF             SGF file to use as reference during test.
  
optional arguments:
  -h, --help            show this help message and exit
  -v VID, --video VID   Filename, or device, as used in cv2.VideoCapture(). Defaults to device "0".
  -b R R, --bounds R R  Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()
  --bf BF               Board finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
  --sf SF               Stones finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
  --failfast            Fail and stop test at first wrong move.
  -m M M, --moves M M   The subsequence of moves to consider in the reference sgf. Provide first and last move number of interest (1-based numeration).Previous moves will be used to initialize the "working" sgf.
```

Examples:
```
$ python3 /Path/To/detectiontest.py --sgf /Path/To/reference_game.sgf -v /Path/To/video.avi
$ python3 /Path/To/detectiontest.py --sgf /Path/To/reference_game.sgf -v /Path/To/video.avi -b 0.1 0.9 --bf BoardFinderAuto --sf SfContours -m 10 20 --failfast
```

## Camkifu - benchmark

For more heavy testing, how about running algorithms on several videos sequentially, and record each results to obtain a global performance view ? 

`CamKifu/test/mains/benchmark.py`

```
usage: benchmark.py dir [-h] [--bf BF] [--sf SF] [--refdir REFDIR]

positional arguments:
  dir              Directory containing the videos on which to run the benchmark.

optional arguments:
  -h, --help       show this help message and exit
  --bf BF          Board finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
  --sf SF          Stones finder class to instantiate at startup. Defaults to configuration defined in cvconf.py
  --refdir REFDIR  Directory containing the reference sgfs associated with each video (matched on filename). Defaults to "dir" if omitted.
```

Examples:
```
$ python3 /Path/To/benchmark.py /Path/To/video_dir/   (the sgf files have to be in that directory as well)
$ python3 /Path/To/benchmark.py /Path/To/video_dir/ --refdir /Path/To/sgf_dir --bf BoardFinderAuto --sf SfContours
```

The current implementation of the report is quite light, feel free to expand the `Report` class to see what you would like, print to a file, etc...