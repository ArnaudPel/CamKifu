#!/bin/sh

rm -rf build dist
cp -r /Users/Kohistan/Developer/PycharmProjects/Golib/src/go /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages
cp -r /Users/Kohistan/Developer/PycharmProjects/Golib/src/gui /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages
cp /Users/Kohistan/Developer/PycharmProjects/Golib/src/golib_conf.py /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages

python setup.py py2app

rm -r /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/go
rm -r /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/gui
rm /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/golib_conf.py