import os
from haar.seedgen import w,h

__author__ = 'Kohistan'

# deprecated code used when trying to train haar detection of a grid

func = "/usr/local/bin/opencv_createsamples"
backgroundfp = "/Users/Kohistan/PycharmProjects/CamKifu/res/training/backgrounds/negatives.dat"
vecfp = "/Users/Kohistan/PycharmProjects/CamKifu/res/training/samples.vec"
seedfp = "/Users/Kohistan/PycharmProjects/CamKifu/res/training/seed.png"


def gen_bg():

    """
    Generates the .dat folder listing the background.
    """
    folder = "/Users/Kohistan/PycharmProjects/CamKifu/res/training/backgrounds"
    # todo implement exception-safe rollback :
    os.chdir(folder)
    os.system("ls > negatives.dat")


def createsamples(seedfp, number, backgroundfp, outfp):

    """
    Call the opencv create sample utility.
    """
    img = " -img " + seedfp
    num = " -num " + str(number)
    bg = " -bg " + backgroundfp
    out = " -vec " + outfp
    command = func + img + num + bg + out + " -maxxangle 0.6 -maxyangle 0 -maxzangle 0.3 -maxidev 100 -bgcolor 0 -bgthresh 0 -w "+str(w)+" -h "+str(h)
    os.system(command)


def display_vec_file(filepath):
    command = func + " -vec "+ filepath + " -w " + str(w) + " -h "+ str(h)
    os.system(command)


if __name__ == '__main__' :
#    gen_bg()
    createsamples(seedfp, 20, backgroundfp, vecfp)
    display_vec_file(vecfp)
