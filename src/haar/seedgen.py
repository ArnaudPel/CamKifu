import cv
import Image
import numpy as np

__author__ = 'Kohistan'

#path = "/Users/Kohistan/PycharmProjects/opencv/bin/opencv_createsamples"
filename = "/Users/Kohistan/PycharmProjects/CamKifu/res/training/seed.png"

# note : macbook 2008 cam size is 640*480  (0.3M pixels)


scale = 4  # must be an even number
w = 19 * scale
h = w



def generate():
    img = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    cv.FillConvexPoly(img, ((0,0),(0,w),(w,h),(h,0)), cv.RGB(0,0,0))
    for i in range(19):
        # horizontal lines
        z = i * scale + (scale/2) -1
        cv.Line(img, ((scale/2), z),(w-(scale/2)-1,z), cv.RGB(255,255,255), 1)
        # vertical lines
        cv.Line(img, (z, (scale/2)),(z, h-(scale/2)-1), cv.RGB(255,255,255), 1)
    cv.SaveImage(filename, img)
    print "created a " + str(w) + "x" + str(h) + " picture: "
    print filename

def maketransparent():
    threshold=100
    dist=5
    img=Image.open(filename).convert('RGBA')
    # np.asarray(img) is read only. Wrap it in np.array to make it modifiable.
    arr=np.array(np.asarray(img))
    r,g,b,a=np.rollaxis(arr,axis=-1)
    mask=((r>threshold)
          & (g>threshold)
          & (b>threshold)
          & (np.abs(r-g)<dist)
          & (np.abs(r-b)<dist)
          & (np.abs(g-b)<dist)
        )
    arr[mask,3]=0
    img=Image.fromarray(arr,mode='RGBA')
    img.save('/Users/Kohistan/PycharmProjects/CamKifu/res/training/seed.png')


if __name__ == '__main__' :
    generate()
#    maketransparent()